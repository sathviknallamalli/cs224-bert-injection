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

# Create a submodel using the first 7 layers of BERT
submodel = BertForMaskedLM.from_pretrained('bert-base-cased', output_hidden_states=True, return_dict = True, num_hidden_layers = 7)

# Define the input sentence with a [MASK] token
input_sentences = ["The man drove the car with a broken " + tokenizer.mask_token + " to the mechanic", 
                   "The man drove the car with a broken " + tokenizer.mask_token + " to the mechanic"]

import numpy as np
def apply_transformation(vector, matrix):
    transformed_vector = torch.mm(matrix, vector)
    return transformed_vector

def unapply_transformation(vector, matrix):
    inv_matrix = torch.tensor(np.linalg.pinv(matrix))
    untransformed_vector = torch.matmul(inv_matrix, vector)
    return untransformed_vector

for sentenceIdx in range(0, 1):
    print("sentence: ", input_sentences[sentenceIdx])

    # obtaining distance matrix 
    with open('data/distance_finals.pkl', 'rb') as f:
        distance_matrices = pickle.load(f)
    #these distance_matricies contain the matrciies for both parse trees for all sentences

    # tokenizing the input sentence
    input = tokenizer(input_sentences[sentenceIdx], return_tensors = "pt")

    print(input.word_ids())

    # Run the input through the submodel to get the hidden states
    output = submodel(**input)

    # identifying location of the masked token
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

    # obtaining the b_matrix for syntactic probe
    import_b = torch.load('bert-base-distance-probes/ptb-prd-BERTbase-rank768.params', map_location='cpu')
    b_matrix = import_b['proj']

    # last layer of hidden states in the sub model (at layer 7)
    hidden_states = output.hidden_states[-1] 
    ground_truth_vector = hidden_states[0][1:-1]


    # setting learning rate & multipling hidden states by b_matrix
    lr = 0.001
    #print sizes
    print("hidden states size: ", hidden_states[0].size())
    print("b_matrix size: ", b_matrix.size())
    transformed_hidden = torch.transpose(apply_transformation(torch.transpose(hidden_states[0], 0, 1), b_matrix), 1, 0)
    #torch.transpose(torch.matmul(b_matrix, torch.transpose(hidden_states[0], 0, 1)), 1, 0)

    #transformed_vector = apply_transformation(input_vector, b_matrix)
    #untransformed_vector = unapply_transformation(transformed_vector, b_matrix)

    # defining a baseline loss function
    #when running this function with the first_hidden_square and every iteration of hidden_square, we should get a loss of 0 - this is confirmed
    def loss_to_original(matrix1, matrix2):
        loss = torch.linalg.norm(matrix1 - matrix2)**2
        #loss = torch.mean(torch.square(matrix1 - matrix2))
        return loss
    
    #running loss to original as our main loss - this should produce like shit results but should low loss values
    #we have really loss values because the magnitude of our hidden vecotrs is much different than the distance matrix
        #multiply H by alpha so that they are similar magnitudes
        #way we do this is - > take norm of H, take norm of D then multipy H by (normD/normH)

      # defining a baseline loss function
    def custom_dual_loss(hidden_square, dist_context, og_hidden_square, theta):
        #theta defines how much we weight stuff
        diffs =  torch.linalg.norm(torch.transpose(hidden_square, 0, 1) - hidden_square, ord = 2, dim = 2)**2
        diffs.requires_grad_(True)
        loss = theta*torch.linalg.norm(og_hidden_square - hidden_square)**2 + torch.mean(torch.square(diffs - dist_context))
        return loss
    
    def custom_dual_scaled_loss(hidden_vectors, hidden_distances, ground_truth_vectors, dist_context, theta):

        #theta defines how much we weight stuff
        hidden_square_scaled = (torch.linalg.norm(dist_context)/torch.linalg.norm(hidden_distances))*hidden_distances
        
        diffs =  torch.linalg.norm(torch.transpose(hidden_square_scaled, 0, 1) - hidden_square_scaled, ord = 2, dim = 2)**2
        diffs.requires_grad_(True)

        #let's untransform the changed hidden vectors here
        untransformed_hidden = torch.transpose(unapply_transformation(torch.transpose(hidden_vectors, 0, 1), b_matrix), 0, 1)
        
        #torch.transpose(torch.matmul(torch.linalg.inv(b_matrix), torch.transpose(hidden_vectors, 0, 1)), 1, 0)
        print(untransformed_hidden)
        print(ground_truth_vectors)
        #loss = theta*torch.linalg.norm(ground_truth_vectors - untransformed_hidden)**2 + (0)*torch.mean(torch.square(diffs - dist_context))
        loss = torch.linalg.norm(ground_truth_vectors - untransformed_hidden)**2 
        print("loss 1 ", torch.linalg.norm(ground_truth_vectors - untransformed_hidden)**2)
        print('loss 2 ', torch.mean(torch.square(diffs - dist_context)))
        return loss


    # removing the padding tokens (first and last hidden states)
    transformed_hidden_no_padding_first = transformed_hidden[1:-1].requires_grad_(True)
    transformed_hidden_no_padding_first.retain_grad()

    # distance matrix for first linguistic context
    distance_first_context = torch.from_numpy(distance_matrices[sentenceIdx])
    distance_first_context = distance_first_context**2
    distance_first_context.requires_grad_(True)

     # This creates a new tensor hidden_square by unsqueezing the transformed_hidden_no_padding tensor along the second dimension,
    #   which adds a new dimension of size 1 at the second position of the tensor. 
    #   This changes the shape of transformed_hidden_no_padding tensor from (seq_len, hidden_size) to (seq_len, 1, hidden_size).
    # The expand call then repeats the tensor along the first and second dimensions, creating a tensor of shape (seq_len, seq_len, hidden_size) 
    #   where each element along the first and second dimensions is a copy of the transformed_hidden_no_padding tensor. 
    #   The resulting tensor hidden_square is used for computing pairwise distances between every pair of hidden states in a sequence.
    #   We later define a diffs matrix that does the actual computation of pairwise distances.
    first_transformation = transformed_hidden_no_padding_first.unsqueeze(1).expand(transformed_hidden_no_padding_first.size()[0], 
                                                                    transformed_hidden_no_padding_first.size()[0], 
                                                                    transformed_hidden_no_padding_first.size()[1])

    initialloss = custom_dual_scaled_loss(transformed_hidden_no_padding_first, first_transformation, ground_truth_vector, distance_first_context, 0.2)

    # training loop for first linguistic context, 10 epochs
    print('Training for first linguistic context')
    difftemp = 0
    loss = 100000000000
    i = 0
    convergence_threshold = 0.009
    maxiters = 10000000
    print('Initial loss: ', initialloss)

    while i < 3:
        i += 1
        # computing pairwise distances between every pair of hidden states in a sequence
        hidden_square = transformed_hidden_no_padding_first.unsqueeze(1).expand(transformed_hidden_no_padding_first.size()[0], 
                                                                        transformed_hidden_no_padding_first.size()[0], 
                                                                        transformed_hidden_no_padding_first.size()[1])

        # computing loss between the computed pariwise distances and the distance matrix for the first linguistic context
        # we can do this because hidden_square is now in the first linguistic context
        #loss = loss_to_original(first_hidden_square, hidden_square)
        #custom_dual_scaled_loss(hidden_square, dist_context, og_hidden_square, theta):
        loss = custom_dual_scaled_loss(transformed_hidden_no_padding_first, hidden_square, ground_truth_vector, distance_first_context, 0.99)
        #loss = custom_dual_loss(hidden_square, distance_first_context, original_hidden_square, 0.5)
        print(loss)

        loss.backward(retain_graph=True)
        if i % 100 == 0:
            print("Loss at step", i, "is", loss.item())
        
        transformed_hidden_no_padding_first.retain_grad()
        transformed_hidden_no_padding_first -= lr*transformed_hidden_no_padding_first.grad.data

        transformed_hidden_no_padding_first.grad.data.zero_() 

    print("convergence loss at step", i, "is", loss.item())
    new_hidden_first = torch.cat((transformed_hidden[0].unsqueeze(0), transformed_hidden_no_padding_first, transformed_hidden[-1].unsqueeze(0)), 0)


    hidden_square = transformed_hidden_no_padding_first.unsqueeze(1).expand(transformed_hidden_no_padding_first.size()[0], 
                                                                    transformed_hidden_no_padding_first.size()[0], 
                                                                    transformed_hidden_no_padding_first.size()[1])
    diffs = torch.linalg.norm(torch.transpose(hidden_square, 0, 1) - hidden_square, ord = 2, dim = 2)**2

    #untransform
    updated_hidden_first = torch.transpose(torch.matmul(torch.linalg.pinv(b_matrix), torch.transpose(new_hidden_first, 0, 1)), 1, 0)
    updated_hidden_first = updated_hidden_first.unsqueeze(0)

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
        new_sentence = input_sentences[sentenceIdx].replace(tokenizer.mask_token, word)
        print(new_sentence)


    # Create a file with 'filename'.pkl in the respective folder, then run the following two commands
    with open("data2/dumpfile" + str(sentenceIdx), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([transformed_hidden_no_padding_first, hidden_square, distance_first_context,
                    new_hidden_first, updated_hidden_first, 
                    output_first_context, logits_first_context, softmax_first_context,
                    mask_word_first_context, top_10_prob_first_context, top_10_first_context], f)
    print("BUZZZ THIS SENTENCE IS DONE")


