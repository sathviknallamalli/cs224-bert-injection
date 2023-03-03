import copy
import torch
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from torch.nn import functional as F
import torch.nn as nn
import pickle
from tqdm import tqdm

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

# obtaining the b_matrix for syntactic probe
import_b = torch.load('bert-base-distance-probes/ptb-prd-BERTbase-rank768.params', map_location='cpu')
b_matrix = import_b['proj']

# last layer of hidden states in the sub model (at layer 5)
hidden_states = output.hidden_states[-1] 

# setting learning rate & multipling hidden states by b_matrix
lr = .00001
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
print('Training for first linguistic context')
for i in tqdm(range(10000)):
    # computing pairwise distances between every pair of hidden states in a sequence
    hidden_square = transformed_hidden_no_padding.unsqueeze(1).expand(transformed_hidden_no_padding.size()[0], 
                                                                      transformed_hidden_no_padding.size()[0], 
                                                                      transformed_hidden_no_padding.size()[1])
    diffs = torch.linalg.norm(torch.transpose(hidden_square, 0, 1) - hidden_square, ord = 2, dim = 2)

    # computing loss between the computed pariwise distances and the distance matrix for the first linguistic context
    # we can do this because hidden_square is now in the first linguistic context
    loss = custom_loss(diffs, distance_first_context)
    loss.backward(retain_graph=True)
    if i % 1000 == 0:
        print("Loss at step", i, "is", loss.item())

    transformed_hidden_no_padding -= lr*transformed_hidden_no_padding.grad.data
    transformed_hidden_no_padding.retain_grad()
    #print(transformed_hidden_no_padding.grad.data)
    transformed_hidden_no_padding.grad.data.zero_() 
    
transformed_hidden_no_padding_second = transformed_hidden[1:-1].requires_grad_(True)
transformed_hidden_no_padding_second.retain_grad()

print('Training for second linguistic context')
for i in tqdm(range(10000)):
    hidden_square = transformed_hidden_no_padding_second.unsqueeze(1).expand(transformed_hidden_no_padding.size()[0], 
                                                                      transformed_hidden_no_padding_second.size()[0], 
                                                                      transformed_hidden_no_padding_second.size()[1])
    diffs = torch.linalg.norm(torch.transpose(hidden_square, 0, 1) - hidden_square, ord = 2, dim = 2)

    loss = custom_loss(diffs, distance_second_context)
    loss.backward(retain_graph=True)
    if i % 1000 == 0:
        print("Loss at step", i, "is", loss.item())

    transformed_hidden_no_padding_second -= lr*transformed_hidden_no_padding_second.grad.data
    transformed_hidden_no_padding_second.retain_grad()
    #print(transformed_hidden_no_padding.grad.data)
    transformed_hidden_no_padding_second.grad.data.zero_()


#add the first and last word from transformed_hidden back to the tensor transformed_hidden1
new_hidden_first = torch.cat((transformed_hidden[0].unsqueeze(0), transformed_hidden_no_padding, transformed_hidden[-1].unsqueeze(0)), 0)

new_hidden_second = torch.cat((transformed_hidden[0].unsqueeze(0), transformed_hidden_no_padding_second, transformed_hidden[-1].unsqueeze(0)), 0)

#updated_hidden = torch.transpose(torch.matmul(torch.linalg.pinv(b_matrix), torch.transpose(new_hidden, 0, 1)), 1, 0)

# *need to add the nullspace version
updated_hidden_first = torch.transpose(torch.matmul(torch.linalg.pinv(b_matrix), torch.transpose(new_hidden_first, 0, 1)), 1, 0)
updated_hidden_first  = updated_hidden_first.unsqueeze(0)

updated_hidden_second = torch.transpose(torch.matmul(torch.linalg.pinv(b_matrix), torch.transpose(new_hidden_second, 0, 1)), 1, 0)
updated_hidden_second  = updated_hidden_second.unsqueeze(0)


oldModuleList = model.bert.encoder.layer
newModuleList = nn.ModuleList()

# Now iterate over all layers, only keeping only the relevant layers.
for i in range(5, 12):
    newModuleList.append(oldModuleList[i])

# create a copy of the model, modify it with the new list, and return
copyOfModel = copy.deepcopy(model)
copyOfModel.bert.encoder.layer = newModuleList

#run the output of the 5 submodel with this other model
#target_bert_model(input_ids, hidden_states=hidden_states)

# print("sanity check")
# print(hidden_states)
# print(transormed_hidden1)
#print(updated_hidden.size())
#print(hidden_states.size())
output_first_context = copyOfModel(inputs_embeds=updated_hidden_first, return_dict = True, output_hidden_states=True)
output_second_context = copyOfModel(inputs_embeds=updated_hidden_second, return_dict = True, output_hidden_states=True)
# print("after split tensor and top words")
#print(output2.hidden_states[-1])

# first context, top 10 words
print('First context, top 10 words')
logits_first_context = output_first_context.logits
softmax_first_context = F.softmax(logits_first_context, dim = -1)
mask_word_first_context = softmax_first_context[0, mask_index, :]
top_10_first_context = torch.topk(mask_word_first_context, 10, dim = 1)[1][0]
top_10_prob_first_context = torch.topk(mask_word_first_context, 10, dim = 1)[0][0]
for i in range(10):
    print(tokenizer.decode([top_10_first_context[i]]), top_10_prob_first_context[i].item())
for token in top_10_first_context:
   word = tokenizer.decode([token])
   new_sentence = input_sentence.replace(tokenizer.mask_token, word)
   print(new_sentence)

# second context, top 10 words
print('Second context, top 10 words')
logits_second_context = output_second_context.logits
softmax_second_context = F.softmax(logits_second_context, dim = -1)
mask_word_second_context = softmax_second_context[0, mask_index, :]
top_10_second_context = torch.topk(mask_word_second_context, 10, dim = 1)[1][0]
top_10_prob_second_context = torch.topk(mask_word_second_context, 10, dim = 1)[0][0]
for i in range(10):
    print(tokenizer.decode([top_10_second_context[i]]), top_10_prob_second_context[i].item())
for token in top_10_second_context:
   word = tokenizer.decode([token])
   new_sentence = input_sentence.replace(tokenizer.mask_token, word)
   print(new_sentence)

 
# Create a file with 'filename'.pkl in the respective folder, then run the following two commands
with open('data/sentence_pickles/cows_first.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([transformed_hidden_no_padding, hidden_square, distance_first_context,
                 new_hidden_first, updated_hidden_first, 
                 output_first_context, logits_first_context, softmax_first_context,
                 mask_word_first_context, top_10_prob_first_context, top_10_first_context], f)


with open('data/sentence_pickles/cows_second.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([transformed_hidden_no_padding, hidden_square, distance_second_context,
                 new_hidden_second, updated_hidden_second, 
                 output_second_context, logits_second_context, softmax_second_context,
                 mask_word_second_context, top_10_prob_second_context, top_10_second_context], f)
    

# # Toggle commenting everything above and below this line to load data. Change the filename accordingly.
# import pickle
# with open('data/sentence_pickles/cows_first.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     transformed_hidden_no_padding, hidden_square, distance_first_context, new_hidden_first, updated_hidden_first, output_first_context, logits_first_context, softmax_first_context, mask_word_first_context, top_10_prob_first_context, top_10_first_context = pickle.load(f)

# with open('data/sentence_pickles/cows_second.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     transformed_hidden_no_padding_second, hidden_square, distance_second_context, new_hidden_second, updated_hidden_second, output_second_context, logits_second_context, softmax_second_context, mask_word_second_context, top_10_prob_second_context, top_10_second_context = pickle.load(f)

# from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

# input_sentence = "They were " + tokenizer.mask_token + " cows"

# for i in range(10):
#     print(tokenizer.decode([top_10_first_context[i]]), top_10_prob_first_context[i].item())
# for token in top_10_first_context:
#    word = tokenizer.decode([token])
#    new_sentence = input_sentence.replace(tokenizer.mask_token, word)
#    print(new_sentence)


# for i in range(10):
#     print(tokenizer.decode([top_10_second_context[i]]), top_10_prob_second_context[i].item())
# for token in top_10_second_context:
#    word = tokenizer.decode([token])
#    new_sentence = input_sentence.replace(tokenizer.mask_token, word)
#    print(new_sentence)

