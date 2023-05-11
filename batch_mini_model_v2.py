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
def apply_transformation(matrix, vector):
    transformed_vector = torch.mm(matrix, vector)
    return transformed_vector

def unapply_transformation(matrix, vector):
    inv_matrix = torch.tensor(np.linalg.pinv(matrix))
    untransformed_vector = torch.matmul(inv_matrix, vector)
    return untransformed_vector
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

def custom_dual_scaled_loss(hidden_vectors, hidden_matrix, ground_truth_vectors, dist_context, theta):

    #let's untransform the changed hidden vectors here
    untransformed_hidden = unapply_transformation(b_matrix, torch.transpose(hidden_vectors, 0, 1)) #768, 12
    untransformed_hidden = torch.transpose(untransformed_hidden, 0, 1) #12,768

    loss1 = torch.mean(torch.square(untransformed_hidden - ground_truth_vectors))

    #theta defines how much we weight stuff
    hidden_square_scaled = (torch.linalg.norm(dist_context)/torch.linalg.norm(hidden_matrix))*hidden_matrix
    
    diffs = torch.linalg.norm(torch.transpose(hidden_square_scaled, 0, 1) - hidden_square_scaled, ord = 2, dim = 2)**2
    diffs.requires_grad_(True)

    loss2 = torch.mean(torch.square(diffs - dist_context))

    loss = theta*loss1 + (1-theta)*loss2
    
    print("loss 1 ", loss1)
    print('loss 2 ', loss2)
    return loss


for sentenceIdx in range(0, 1):
    print("sentence: ", input_sentences[sentenceIdx])

    # obtaining distance matrix 
    with open('data/distance_finals.pkl', 'rb') as f:
        distance_matrices = pickle.load(f)

    input = tokenizer(input_sentences[sentenceIdx], return_tensors = "pt")
    #print(input.word_ids())
    output = submodel(**input)
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

    import_b = torch.load('bert-base-distance-probes/ptb-prd-BERTbase-rank768.params', map_location='cpu')
    b_matrix = import_b['proj']

    # last layer of hidden states in the sub model (at layer 7)
    hidden_states = output.hidden_states[-1] 
    compare_to_bert_space_vectors = hidden_states[0][1:-1] #12,768

    transformed_hidden = apply_transformation(b_matrix, torch.transpose(compare_to_bert_space_vectors, 0, 1)) #768,12
    transformed_hidden = torch.transpose(transformed_hidden, 0, 1) #12,768

    transformed_hidden = transformed_hidden.requires_grad_(True)
    transformed_hidden.retain_grad()

    # distance matrix for first linguistic context
    distance_first_context = torch.from_numpy(distance_matrices[sentenceIdx])
    distance_first_context = distance_first_context**2
    distance_first_context.requires_grad_(True)

    theta = 0.8

    first_transformation_matrix = transformed_hidden.unsqueeze(1).expand(transformed_hidden.size()[0], 
                                                                    transformed_hidden.size()[0], 
                                                                    transformed_hidden.size()[1]) #12,12,768

    initialloss = custom_dual_scaled_loss(transformed_hidden, first_transformation_matrix, compare_to_bert_space_vectors, distance_first_context, theta)
    print('Initial loss: ', initialloss)

    # training loop for first linguistic context, 10 epochs
    loss = 100000000000
    i = 0

    # setting learning rate & multipling hidden states by b_matrix
    lr = 0.001

    while i < 4:
        i += 1
        # computing pairwise distances between every pair of hidden states in a sequence
        hidden_square_matrix = transformed_hidden.unsqueeze(1).expand(transformed_hidden.size()[0], 
                                                                        transformed_hidden.size()[0], 
                                                                        transformed_hidden.size()[1])

        loss = custom_dual_scaled_loss(transformed_hidden, hidden_square_matrix, compare_to_bert_space_vectors, distance_first_context, theta)
        print(loss)

        loss.backward(retain_graph=True)
        if i % 100 == 0:
            print("Loss at step", i, "is", loss.item())
        
        transformed_hidden.retain_grad()
        transformed_hidden -= lr*transformed_hidden.grad.data

        transformed_hidden.grad.data.zero_() 



    print("convergence loss at step", i, "is", loss.item())
    new_hidden_first = torch.cat((transformed_hidden[0].unsqueeze(0), transformed_hidden, transformed_hidden[-1].unsqueeze(0)), 0)

    #untransform
    updated_hidden_first = torch.transpose(torch.matmul(torch.linalg.pinv(b_matrix), torch.transpose(new_hidden_first, 0, 1)), 1, 0)
    updated_hidden_first = updated_hidden_first.unsqueeze(0)

    oldModuleList = model.bert.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keeping only the relevant layers.
    for i in range(7, 12):
        newModuleList.append(oldModuleList[i])

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
    """ with open("data2/dumpfile" + str(sentenceIdx), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([transformed_hidden, hidden_square, distance_first_context,
                    new_hidden_first, updated_hidden_first, 
                    output_first_context, logits_first_context, softmax_first_context,
                    mask_word_first_context, top_10_prob_first_context, top_10_first_context], f) """
    print("BUZZZ THIS SENTENCE IS DONE")


