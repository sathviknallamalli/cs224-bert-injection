import copy
import torch
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from torch.nn import functional as F
import torch.nn as nn
import pickle
from tqdm import tqdm
from sympy import Matrix
# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')

# Create a submodel using the first 5 layers of BERT
submodel = BertForMaskedLM.from_pretrained('bert-base-cased', output_hidden_states=True, return_dict = True, num_hidden_layers = 7)
# Define the input sentence with a [MASK] token
input_sentence = "The man drove the car with a broken " + tokenizer.mask_token + " to the mechanic"
#input_sentence = "The hospital admitted the patient with " + tokenizer.mask_token + " because she required intensive care"
#input_sentence = "the author that likes the security guards " +  tokenizer.mask_token  + " during the show"
# obtaining distance matrix 
with open('data/distance_cars2.pkl', 'rb') as f:
    distance_matrices = pickle.load(f)
print(len(distance_matrices))

# tokenizing the input sentence
input = tokenizer(input_sentence, return_tensors = "pt")
print(input.word_ids())

# Run the input through the submodel to get the hidden states
output = submodel(**input)

# identifying location of the masked token
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

# obtaining the b_matrix for syntactic probe
import_b = torch.load('bert-base-distance-probes/ptb-prd-BERTbase-rank768.params', map_location='cpu')
b_matrix = import_b['proj']
rank = torch.linalg.matrix_rank(b_matrix)
print("Rank:", rank)
A = Matrix(b_matrix)
print(A.nullspace())

# last layer of hidden states in the sub model (at layer 5)
hidden_states = output.hidden_states[-1] 

# setting learning rate & multipling hidden states by b_matrix 
#TODO MAKE SURE TO CHANGE THIS  
lr = 0.001
transformed_hidden = torch.transpose(torch.matmul(b_matrix, torch.transpose(hidden_states[0], 0, 1)), 1, 0)
#transformed_hidden = hidden_states[0]
# splitting each word's hidden state into a separate vector

# removing the padding tokens (first and last hidden states)
transformed_hidden_no_padding_first = transformed_hidden[1:-1].requires_grad_(True)
transformed_hidden_no_padding_first.retain_grad()

# distance matrix for first linguistic context
distance_first_context = torch.from_numpy(distance_matrices[1])
distance_first_context = distance_first_context**2
distance_first_context.requires_grad_(True)


#initial loss
hidden_square = transformed_hidden_no_padding_first.unsqueeze(1).expand(transformed_hidden_no_padding_first.size()[0], 
                                                                      transformed_hidden_no_padding_first.size()[0], 
                                                                      transformed_hidden_no_padding_first.size()[1])
diffs = torch.linalg.norm(torch.transpose(hidden_square, 0, 1) - hidden_square, ord = 2, dim = 2)**2
diffs.requires_grad_(True)
#initial_loss = custom_loss(diffs, distance_first_context)
#target_loss = .4 * initial_loss

print("initial diffs")
print(diffs)
print(distance_first_context)


optimizer = torch.optim.SGD([transformed_hidden_no_padding_first], lr=lr)


# defining a baseline loss function
def custom_loss(matrix_1, matrix_2):

    #loss = F.l1_loss(matrix_1, matrix_2)
    #return loss
    #loss = torch.sum(torch.abs(matrix_1 - matrix_2))

    loss = torch.mean(torch.square(matrix_1 - matrix_2))

    #print(torch.linalg.norm(matrix_1[1] - matrix_1[-2])**2)
    return loss

# training loop for first linguistic context, 10 epochs
print('Training for first linguistic context')

difftemp = 0
minloss = 100000
i = 0
convergence_threshold = 0.009
maxiters = 10000000
while loss > initialloss * 0.025:
    i += 1
    # computing pairwise distances between every pair of hidden states in a sequence
    hidden_square = transformed_hidden_no_padding_first.unsqueeze(1).expand(transformed_hidden_no_padding_first.size()[0], 
                                                                      transformed_hidden_no_padding_first.size()[0], 
                                                                      transformed_hidden_no_padding_first.size()[1])
    diffs = torch.linalg.norm(torch.transpose(hidden_square, 0, 1) - hidden_square, ord = 2, dim = 2)**2
    difftemp = diffs
    #we want diffs to get closer to distance matrix

    # computing loss between the computed pariwise distances and the distance matrix for the first linguistic context
    # we can do this because hidden_square is now in the first linguistic context
    loss = custom_loss(diffs, distance_first_context)
    minloss = min(minloss, loss)
    #print(loss)

    loss.backward(retain_graph=True)
    if i % 100 == 0:
        print("Loss at step", i, "is", loss.item())
        #print(transformed_hidden_no_padding_first.grad.data)
    transformed_hidden_no_padding_first.retain_grad()
    #print(torch.norm(transformed_hidden_no_padding_first.grad.data))
    if torch.norm(transformed_hidden_no_padding_first.grad.data) < convergence_threshold:
        print(f'Converged in {i} iterations')
        break
    transformed_hidden_no_padding_first -= lr*transformed_hidden_no_padding_first.grad.data
    transformed_hidden_no_padding_first.grad.data.zero_() 

print("compare here")
print(difftemp)
print(distance_first_context)

#add the first and last word from transformed_hidden back to the tensor transformed_hidden1
new_hidden_first = torch.cat((transformed_hidden[0].unsqueeze(0), transformed_hidden_no_padding_first, transformed_hidden[-1].unsqueeze(0)), 0)

#this is the old stuff, where we didnt apply the null space
updated_hidden_first = torch.transpose(torch.matmul(torch.linalg.pinv(b_matrix), torch.transpose(new_hidden_first, 0, 1)), 1, 0)
updated_hidden_first = updated_hidden_first.unsqueeze(0)

""" #this is a new null space attempt
reg_param = 1e-7
pinv_b = torch.pinverse(b_matrix, rcond=reg_param)
nullspace_basis = torch.zeros((768, 0))
if pinv_b.shape[1] < 768:
    nullspace_basis = torch.transpose(torch.linalg.qr(pinv_b, mode='reduced')[0][:, pinv_b.shape[1]:], 0, 1)
projection_matrix = torch.eye(768) - torch.matmul(nullspace_basis, nullspace_basis.transpose(0, 1))
projected_hidden = torch.matmul(projection_matrix, transformed_hidden.transpose(0, 1)).transpose(0, 1)
original_hidden = torch.matmul(pinv_b, projected_hidden.transpose(0, 1)).transpose(0, 1)

# create new hidden state
transformed_hidden_no_padding_first = transformed_hidden[1:-1].requires_grad_(True)
new_hidden_first = torch.cat((transformed_hidden[0].unsqueeze(0), transformed_hidden_no_padding_first, transformed_hidden[-1].unsqueeze(0)), 0)
updated_hidden_first = torch.cat((original_hidden[0].unsqueeze(0), original_hidden[1:-1], original_hidden[-1].unsqueeze(0)), 0)
updated_hidden_first = original_hidden.unsqueeze(0) """

oldModuleList = model.bert.encoder.layer
newModuleList = nn.ModuleList()

# Now iterate over all layers, only keeping only the relevant layers.
for i in range(7, 12):
    newModuleList.append(oldModuleList[i])

# create a copy of the model, modify it with the new list, and return
copyOfModel = copy.deepcopy(model)
copyOfModel.bert.encoder.layer = newModuleList

output_first_context = copyOfModel(inputs_embeds = updated_hidden_first, return_dict = True, output_hidden_states = True)

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
