

import copy
import torch
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from torch.nn import functional as F
import torch.nn as nn
import pickle
from tqdm import tqdm
import numpy as np
import math

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')


#CREATE A FILENAME FOR THE REPORT
report = open("tester.txt", "w")

with open('john_sentences.txt', 'r') as file:
    # Read lines from the file and store them in a list
    input_sentences = file.read().splitlines()  #116

# obtaining distance matrix 
with open('distance_john', 'rb') as f:
    distance_matrices = pickle.load(f)
    
distance_dict = {}  #58
for i in range(len(input_sentences)):
    if input_sentences[i] not in distance_dict:
        distance_dict[input_sentences[i]] = []
    distance_dict[input_sentences[i]].append(distance_matrices[i])

def apply_transformation(matrix, vector):
    transformed_vector = torch.mm(torch.transpose(matrix, 0, 1), vector)
    return transformed_vector
    

valid_words = []
invalid_words = []

sentences = list(input_sentences)  #58
sentences = [sentences[i] for i in range(0, len(sentences), 2)]
print(sentences)

import json
with open("dataset.json") as f:
    data = json.load(f)

for sent in data:
    attachment1_words = sent["attachment1_words"]
    attachment2_words = sent["attachment2_words"]

    min_length = min(len(attachment1_words), len(attachment2_words))
    attachment1_words = attachment1_words[:min_length]
    attachment2_words = attachment2_words[:min_length]

    valid_words.append(attachment1_words)
    invalid_words.append(attachment2_words)


def b_loss(hidden_vectors, dist_context, theta, original_hidden_vectors, A_mask):
    transformed_hidden = apply_transformation(b_matrix, torch.transpose(hidden_vectors, 0, 1)) #768,12
    transformed_hidden = torch.transpose(transformed_hidden, 0, 1) #12,768
    hidden_matrix = transformed_hidden.unsqueeze(1).expand(transformed_hidden.size()[0], 
                                                                    transformed_hidden.size()[0], 
                                                                    transformed_hidden.size()[1]) #12,12,768

    hidden_square_scaled = (torch.linalg.norm(dist_context)/torch.linalg.norm(hidden_matrix))*hidden_matrix

    diffs = torch.linalg.norm(torch.transpose(hidden_square_scaled, 0, 1) - hidden_square_scaled, ord = 2, dim = 2)**2
    diffs.requires_grad_(True)
    loss2 = torch.mean(torch.from_numpy(A_mask).type(torch.DoubleTensor) * torch.square(diffs - dist_context))
    loss1 = torch.norm(torch.square(hidden_vectors - original_hidden_vectors), 2)

    loss = theta*loss1 + (1-theta)*loss2
    return loss

for layer in range(11, 12):
    report.write(f'LAYER is {layer} \n')
    
    submodel = BertForMaskedLM.from_pretrained('bert-base-cased', output_hidden_states=True, return_dict = True, num_hidden_layers = layer)

    for rank in [ 16]:
        for theta in [0.3]:
            report.write(f'\nNEW RUN: layer {layer} rank {rank} theta {theta}\n')
            print(f'\nNEW RUN: layer {layer} rank {rank} theta {theta}\n')
            increase_score = 0
            decrease_score = 0
            for sent_index in range(len(sentences)):
                sent = sentences[sent_index]
            
                print(sent_index, sent)
                print("index " + str(sent_index) + " " + "this sent")
                input = tokenizer(sent, return_tensors = "pt")
                orig_output = model(**input)
                mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
                orig_logits = orig_output.logits
                orig_softmax = F.softmax(orig_logits, dim = -1)
                orig_mask_word = orig_softmax[0, mask_index, :]
            
            
                #made a dictionary of all the original BERT model words and predictions
                orig_predictions = {}

                orig_values_second_dimension = orig_mask_word[0, :]
                orig_num_predictions = orig_mask_word[0, :].shape[0]

                orig_all_words = torch.topk(orig_mask_word, orig_num_predictions, dim = 1)[1][0]
                orig_all_probabilites = torch.topk(orig_mask_word, orig_num_predictions, dim = 1)[0][0]
                orig_counter = 0
                for token in orig_all_words:
                    word = tokenizer.decode([token])
                    orig_predictions[word] = orig_all_probabilites[orig_counter].item()
                    orig_counter += 1
                    
                for i in range(2):
                    report.write(sent + "\n")
                    input = tokenizer(sent, return_tensors = "pt")
                    output = submodel(**input)
                    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

                    # report.write(f"Associated Distance Matrices: {distance_dict[sent]}\n")

                    A_mask = np.not_equal(distance_dict[sent][0], distance_dict[sent][1]).astype(int)

                    b_matrix_path = f"probes/r{str(rank)}/{str(layer)}/predictor_random.params"
                    import_b = torch.load(b_matrix_path, map_location='cpu')
                    b_matrix = import_b['proj']
                    # print(import_b)

                    # last layer of hidden states in the sub model
                    hidden_states = output.hidden_states[-1] 
                    optimized_hidden_vectors = hidden_states[0][1:-1] #12,768 
                    og_hidden_vectors = optimized_hidden_vectors.clone()
                    optimized_hidden_vectors = optimized_hidden_vectors.requires_grad_(True)
                    optimized_hidden_vectors.retain_grad()

                    # distance matrix for first linguistic context
                    distance_first_context = torch.from_numpy(distance_dict[sent][i])
                    distance_first_context = distance_first_context**2
                    distance_first_context.requires_grad_(True)

                    #report.write(f'theta is {theta} \n')
                    initialloss = b_loss(optimized_hidden_vectors, distance_first_context, theta, og_hidden_vectors, A_mask)
                
                    # report.write(f"Initial loss: {initialloss}\n")

                    # training loop for first linguistic context, 10 epochs
                    loss = 100000000000
                    iters = 0

                    # setting learning rate & multipling hidden states by b_matrix
                    lr = 0.001
                    # report.write(f'learning rate is {lr} \n')

                    for i in tqdm(range(500)):
                    #while iters < 500:
                        iters += 1
                        # computing pairwise distances between every pair of hidden states in a sequence
                        loss = b_loss(optimized_hidden_vectors, distance_first_context, theta, og_hidden_vectors, A_mask)

                        loss.backward(retain_graph=True)
                        # if i % 100 == 0:
                        #     report.write(f'Loss at step {i} is {loss.item()}.\n')
                        
                        optimized_hidden_vectors.retain_grad()
                        optimized_hidden_vectors -= lr*optimized_hidden_vectors.grad.data

                        optimized_hidden_vectors.grad.data.zero_() 

                    new_hidden_first = torch.cat((hidden_states[0][0].unsqueeze(0), optimized_hidden_vectors, hidden_states[0][-1].unsqueeze(0)), 0).unsqueeze(0)
                    oldModuleList = model.bert.encoder.layer
                    newModuleList = nn.ModuleList()

                    # Now iterate over all layers, only keeping only the relevant layers.
                    for bert_layers in range(layer, 12):
                        newModuleList.append(oldModuleList[bert_layers])

                    copyOfModel = copy.deepcopy(model)
                    copyOfModel.bert.encoder.layer = newModuleList

                    output_first_context = copyOfModel(inputs_embeds = new_hidden_first, return_dict = True, output_hidden_states = True)
                    
                    # print('Top 10 words')
                    logits_first_context = output_first_context.logits
                    softmax_first_context = F.softmax(logits_first_context, dim = -1)
                    mask_word_first_context = softmax_first_context[0, mask_index, :]                    

                    # print("BUZZZ THIS SENTENCE IS DONE")


                    #here, the score metric u r creating will go here, using the all_predictions dictionary and the 5 words for each 
                    #context i've written

                    #made a dictionary of all the context 1 words and their probabilitiess
                    all_predictions = {}
                    values_second_dimension = mask_word_first_context[0, :]
                    num_predictions = values_second_dimension.shape[0]

                    all_words = torch.topk(mask_word_first_context, num_predictions, dim = 1)[1][0]
                    all_probabilites = torch.topk(mask_word_first_context, num_predictions, dim = 1)[0][0]
                    counter = 0
                    for token in all_words:
                        word = tokenizer.decode([token])
                        all_predictions[word] = all_probabilites[counter].item()
                        counter += 1
                    

                    

                    

                    allowed_words = []
                    disallowed_words = []
                    if i == 0:  #first context
                        allowed_words = valid_words[sent_index] #these words are for the first context
                        disallowed_words = invalid_words[sent_index]
                    else:
                        allowed_words = invalid_words[sent_index] #these words are for the second context
                        disallowed_words = valid_words[sent_index]

                    minVal = min(all_predictions.values())
                    for word_index in range(len(allowed_words)):
                        model_allow = 0 
                        model_disallow = 0
                        orig_allow = 0
                        orig_disallow = 0

                        if allowed_words[word_index] in all_predictions:
                            model_allow = all_predictions[allowed_words[word_index]]
                        if allowed_words[word_index] in orig_predictions:
                            orig_allow = orig_predictions[allowed_words[word_index]]

                        if disallowed_words[word_index] in all_predictions:
                            model_disallow = all_predictions[disallowed_words[word_index]]
                        if disallowed_words[word_index] in orig_predictions:
                            orig_disallow = orig_predictions[disallowed_words[word_index]]

                        if model_allow != 0 or orig_allow != 0:
                            increase_score += (math.log(model_allow) - math.log(orig_allow))
                            #write the original and model probabilities for the allowed words
                            report.write(f'orig_allow is {orig_allow} and model_allow is {model_allow}.\n')
                        if model_disallow != 0 or orig_disallow != 0:
                            decrease_score += (math.log(orig_disallow) - math.log(model_disallow))
                            report.write(f'orig_allow is {orig_disallow} and model_allow is {model_disallow}.\n')
            print(1.0*increase_score/len(sentences) + 1.0*decrease_score/len(sentences))
            report.write(f'Average increase, decrease scores is {1.0*increase_score/len(sentences) + 1.0*decrease_score/len(sentences)}.\n')

                #score metric: average increase in log probabilities for the allowed words plus the average decrease in log probabilities for the disallowed words.
report.close()