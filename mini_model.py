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

lr = .010
transormed_hidden = torch.transpose(torch.matmul(pretend_b,torch.transpose(hidden_states[0],0,1)), 1, 0)
split_hvecs = [transormed_hidden[i].requires_grad_(True) for i in range(0, transormed_hidden.size()[0])]
for hvec in split_hvecs:
    hvec.retain_grad()

distance_matrix = torch.from_numpy(distance_matrices[3])
word_idxs = input.word_ids()
full_loss = 0

def custom_loss(hvec_1, hvec_2, idx1, idx2):
    
    #loss = torch.abs(torch.mean((hvec_1 - hvec_2)**2) - distance_matrix[word_idxs[idx1]-1][word_idxs[idx2]-1])
    
    #if idx1 == 4 and idx2  == 2:
        #print(loss)
        #print(distance_matrix[word_idxs[idx1]-1][word_idxs[idx2]-1])
    loss = torch.mean((hvec_1 - hvec_2)**2)

    return loss

for i in range(10):
    
    full_loss = 0.0

    for idx1, hvec_1 in list(enumerate(split_hvecs))[1:-1]:
        for idx2, hvec_2 in list(enumerate(split_hvecs))[1:-1]:

            if idx1 != idx2:
                full_loss += custom_loss(hvec_1, hvec_2, idx1, idx2)
    
    full_loss = torch.mean(full_loss)
    full_loss.backward(retain_graph=True)

    for idx, hvec in list(enumerate(split_hvecs))[1:-1]:
        split_hvecs[idx] = hvec -   lr*hvec.grad.data #IS THIS PLUS IDK PLS HELP I THINK ITS MINUS  
        split_hvecs[idx].retain_grad()
        hvec.grad.data.zero_()

new_hidden = torch.stack(split_hvecs)


print(new_hidden)
print(transormed_hidden)
#print(new_hidden.size())


"""
  def forward(self, batch):
     Computes all n^2 pairs of difference scores 
    for each sentence in a batch.
    Note that due to padding, some distances will be non-zero for pads.
    Computes (h_i-h_j)^TA(h_i-h_j) for all i,j
    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of scores of shape (batch_size, max_seq_len, max_seq_len)
    
    batchlen, seqlen, rank = batch.size()
    batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)
    diffs = (batch_square - batch_square.transpose(1,2)).view(batchlen*seqlen*seqlen, rank)
    psd_transformed = torch.matmul(diffs, self.proj).view(batchlen*seqlen*seqlen,1,rank)
    dists = torch.bmm(psd_transformed, diffs.view(batchlen*seqlen*seqlen, rank, 1))
    dists = dists.view(batchlen, seqlen, seqlen)
    return dists
"""