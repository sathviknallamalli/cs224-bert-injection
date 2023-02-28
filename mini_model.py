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
submodel = BertForMaskedLM.from_pretrained('bert-base-cased', output_hidden_states=True, return_dict = True, num_hidden_layers=5)
# Define the input sentence with a [MASK] token
input_sentence = "The burglar blew open the safe with the " + tokenizer.mask_token


# let's just say for now, this is the third sentence in our corpus, so we are using the third distance matrix that we calcualted

with open('/Users/aakritilakshmanan/cs224-bert-injection/data/distance.pkl', 'rb') as f:
    distance_matrices = pickle.load(f)

input = tokenizer(input_sentence, return_tensors = "pt")

# Run the input through the submodel to get the hidden states
output = submodel(**input)

mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

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

# Get the probabilities for the top 5 probable words for the mask
""" mask_hidden_state = hidden_states[0][mask_position]
top_k_indices = torch.topk(mask_hidden_state, k=5).indices.tolist()
top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

# Print the top 5 probable words for the mask
print("Top 5 probable words for the mask:")
for token in top_k_tokens:
    print(token) """

pretend_b = torch.ones([768,768])

hidden_states = output.hidden_states[-1] 

lr = 0.001
transormed_hidden = torch.transpose(torch.matmul(pretend_b,torch.transpose(hidden_states[0],0,1)), 1, 0)
split_hvecs = [transormed_hidden[i].requires_grad_(True) for i in range(0, transormed_hidden.size()[0])]
for hvec in split_hvecs:
    hvec.retain_grad()

distance_matrix = torch.from_numpy(distance_matrices[3])
word_idxs = input.word_ids()
full_loss = 0
loss = nn.MSELoss()

def custom_loss(hvec_1, hvec_2, idx1, idx2):
    loss = torch.subtract(torch.mean((hvec_1 - hvec_2)**2), distance_matrix[word_idxs[idx1]-1][word_idxs[idx2]-1])
    #need to think about how the hvec terms factor into the distance terms

    #loss = torch.mean((hvec_1 - hvec_2)**2)

    return loss

for idx1, hvec_1 in list(enumerate(split_hvecs))[1:-1]:
    for idx2, hvec_2 in  list(enumerate(split_hvecs))[1:-1]:

        if idx1 != idx2:
            full_loss += custom_loss(hvec_1, hvec_2, idx1, idx2)

full_loss.backward()

for idx, hvec in  list(enumerate(split_hvecs))[1:-1]:
    split_hvecs[idx] = hvec -  lr*hvec.grad.data #IS THIS PLUS IDK PLS HELP I THINK ITS MINUS  
    hvec.grad.data.zero_()

new_hidden = torch.stack(split_hvecs)


print(new_hidden)
print(transormed_hidden)
print(new_hidden.size())
