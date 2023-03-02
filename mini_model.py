import copy
import torch
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from torch.nn import functional as F
import torch.nn as nn
import pickle

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')

# Create a submodel using the first 5 layers of BERT
submodel = BertForMaskedLM.from_pretrained('bert-base-cased', output_hidden_states=True, return_dict = True, num_hidden_layers = 5)

# Define the input sentence with a [MASK] token
input_sentence = "They were " + tokenizer.mask_token + " cows"

# obtaining distance matrix 
with open('data/distance.pkl', 'rb') as f:
    distance_matrices = pickle.load(f)

# tokenizing the input sentence
input = tokenizer(input_sentence, return_tensors = "pt")

# Run the input through the submodel to get the hidden states
output = submodel(**input)

# identifying location of the masked token
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

# printing out top 10 predicted words and their probabilities (for submodel after 5 layers)
logits = output.logits
softmax = F.softmax(logits, dim = -1)
mask_word = softmax[0, mask_index, :]
top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
top_10_prob = torch.topk(mask_word, 10, dim = 1)[0][0]
for i in range(10):
    print(tokenizer.decode([top_10[i]]), top_10_prob[i].item())
for token in top_10:
   word = tokenizer.decode([token])
   new_sentence = input_sentence.replace(tokenizer.mask_token, word)
   print(new_sentence)

# obtaining the b_matrix for syntactic probe
import_b = torch.load('bert-base-distance-probes/ptb-prd-BERTbase-rank768.params', map_location='cpu')
b_matrix = import_b['proj']

# last layer of hidden states in the sub model (at layer 5)
hidden_states = output.hidden_states[-1] 

# setting learning rate & multipling hidden states by b_matrix
lr = .010
transformed_hidden = torch.transpose(torch.matmul(b_matrix, torch.transpose(hidden_states[0], 0, 1)), 1, 0)

# splitting each word's hidden state into a separate vector
split_hvecs = [transformed_hidden[i].requires_grad_(True) for i in range(0, transformed_hidden.size()[0])]
for hvec in split_hvecs:
    hvec.retain_grad()

# defining a baseline loss function
def custom_loss(matrix_1, matrix_2):
    loss = torch.sum(torch.abs(matrix_1 - matrix_2))
    return loss

# removing the padding tokens (first and last hidden states)
transformed_hidden_no_padding = transformed_hidden[1:-1].requires_grad_(True)
transformed_hidden_no_padding.retain_grad()

# This creates a new tensor hidden_square by unsqueezing the transformed_hidden_no_padding tensor along the second dimension,
#   which adds a new dimension of size 1 at the second position of the tensor. 
#   This changes the shape of transformed_hidden_no_padding tensor from (seq_len, hidden_size) to (seq_len, 1, hidden_size).
# The expand call then repeats the tensor along the first and second dimensions, creating a tensor of shape (seq_len, seq_len, hidden_size) 
#   where each element along the first and second dimensions is a copy of the transformed_hidden_no_padding tensor. 
#   The resulting tensor hidden_square is used for computing pairwise distances between every pair of hidden states in a sequence.
#   We later define a diffs matrix that does the actual computation of pairwise distances.
hidden_square = transformed_hidden_no_padding.unsqueeze(1).expand(transformed_hidden_no_padding.size()[0], 
                                                                  transformed_hidden_no_padding.size()[0], 
                                                                  transformed_hidden_no_padding.size()[1])

# distance matrix for first linguistic context
distance_first_context = torch.from_numpy(distance_matrices[16])
distance_first_context.requires_grad_(True)

# distance matrix for second linguistic context
distance_second_context = torch.from_numpy(distance_matrices[17])
distance_second_context.requires_grad_(True)

# training loop for first linguistic context, 10 epochs
for i in range(10):
    # computing pairwise distances between every pair of hidden states in a sequence
    hidden_square = transformed_hidden_no_padding.unsqueeze(1).expand(transformed_hidden_no_padding.size()[0], 
                                                                      transformed_hidden_no_padding.size()[0], 
                                                                      transformed_hidden_no_padding.size()[1])
    diffs = torch.linalg.norm(torch.transpose(hidden_square, 0, 1) - hidden_square, ord = 2, dim = 2)

    # computing loss between the computed pariwise distances and the distance matrix for the first linguistic context
    # we can do this because hidden_square is now in the first linguistic context
    loss = custom_loss(diffs, distance_first_context)
    loss.backward(retain_graph=True)

    transformed_hidden_no_padding -= lr*transformed_hidden_no_padding.grad.data
    transformed_hidden_no_padding.retain_grad()
    print(transformed_hidden_no_padding.grad.data)
    transformed_hidden_no_padding.grad.data.zero_()
    

transformed_hidden_no_padding = transformed_hidden[1:-1].requires_grad_(True)
transformed_hidden_no_padding.retain_grad()

for i in range(10):
    hidden_square = transformed_hidden_no_padding.unsqueeze(1).expand(transformed_hidden_no_padding.size()[0], 
                                                                      transformed_hidden_no_padding.size()[0], 
                                                                      transformed_hidden_no_padding.size()[1])
    diffs = torch.linalg.norm(torch.transpose(hidden_square, 0, 1)- hidden_square, ord = 2, dim = 2)

    loss = custom_loss(diffs, distance_second_context)
    loss.backward(retain_graph=True)

    transformed_hidden_no_padding -= lr*transformed_hidden_no_padding.grad.data
    transformed_hidden_no_padding.retain_grad()
    print(transformed_hidden_no_padding.grad.data)
    transformed_hidden_no_padding.grad.data.zero_()



