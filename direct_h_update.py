import copy
import torch
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from torch.nn import functional as F
import torch.nn as nn
import pickle
from tqdm import tqdm
import numpy as np

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')


#CREATE A FILENAME FOR THE REPORT
report = open("car.txt", "w")


input_sentences = ["The man drove the car with a broken " + tokenizer.mask_token + " to the mechanic", 
                   "The man drove the car with a broken " + tokenizer.mask_token + " to the mechanic"]

# obtaining distance matrix 
with open('/Users/aditya/Downloads/unscaled_raw_distances.pkl', 'rb') as f:
    distance_matrices = pickle.load(f)

distance_dict = {}
for i in range(len(input_sentences)):
    if input_sentences[i] not in distance_dict:
        distance_dict[input_sentences[i]] = []
    distance_dict[input_sentences[i]].append(distance_matrices[i])


def apply_transformation(matrix, vector):
    transformed_vector = torch.mm(matrix, vector)
    return transformed_vector

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

for layer in range(0,11):
    
    submodel = BertForMaskedLM.from_pretrained('bert-base-cased', output_hidden_states=True, return_dict = True, num_hidden_layers = layer)

    for rank in [4, 8, 16, 32, 64]:
        for theta in [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.6, 1]:

            report.write(f'\nNEW RUN OF CODE\n')
            sent = "The man drove the car with a broken [MASK] to the mechanic"

            for i in range(2):
                report.write(sent + "\n")

                input = tokenizer(sent, return_tensors = "pt")
                output = submodel(**input)
                mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

                report.write(f"Associated Distance Matrices: {distance_dict[sent]}\n")

                A_mask = np.not_equal(distance_dict[sent][0], distance_dict[sent][1]).astype(int)

                b_matrix_path = f"/Users/adityatadimeti/Downloads/probes/r{str(rank)}/{str(layer)}/predictior.params"
                import_b = torch.load(b_matrix_path, map_location='cpu')
                b_matrix = import_b['proj']
                print(import_b)

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

                report.write(f'theta is {theta} \n')
                initialloss = b_loss(optimized_hidden_vectors, distance_first_context, theta, og_hidden_vectors, A_mask)
            
                report.write(f"Initial loss: {initialloss}\n")

                # training loop for first linguistic context, 10 epochs
                loss = 100000000000
                i = 0

                # setting learning rate & multipling hidden states by b_matrix
                lr = 0.001
                report.write(f'learning rate is {lr} \n')

                while i < 500:
                    i += 1
                    # computing pairwise distances between every pair of hidden states in a sequence
                    loss = b_loss(optimized_hidden_vectors, distance_first_context, theta, og_hidden_vectors, A_mask)

                    loss.backward(retain_graph=True)
                    if i % 100 == 0:
                        report.write(f'Loss at step {i} is {loss.item()}.\n')
                    
                    optimized_hidden_vectors.retain_grad()
                    optimized_hidden_vectors -= lr*optimized_hidden_vectors.grad.data

                    optimized_hidden_vectors.grad.data.zero_() 

                new_hidden_first = torch.cat((hidden_states[0][0].unsqueeze(0), optimized_hidden_vectors, hidden_states[0][-1].unsqueeze(0)), 0).unsqueeze(0)
                oldModuleList = model.bert.encoder.layer
                newModuleList = nn.ModuleList()

                # Now iterate over all layers, only keeping only the relevant layers.
                for i in range(layer, 12):
                    newModuleList.append(oldModuleList[i])

                copyOfModel = copy.deepcopy(model)
                copyOfModel.bert.encoder.layer = newModuleList

                output_first_context = copyOfModel(inputs_embeds = new_hidden_first, return_dict = True, output_hidden_states = True)
                
                print('Top 10 words')
                logits_first_context = output_first_context.logits
                softmax_first_context = F.softmax(logits_first_context, dim = -1)
                mask_word_first_context = softmax_first_context[0, mask_index, :]
                top_10_first_context = torch.topk(mask_word_first_context, 10, dim = 1)[1][0]
                top_10_prob_first_context = torch.topk(mask_word_first_context, 10, dim = 1)[0][0]
                for i in range(10):
                    report.write(f"{tokenizer.decode([top_10_first_context[i]])} {top_10_prob_first_context[i].item()} \n")
                    print(tokenizer.decode([top_10_first_context[i]]), top_10_prob_first_context[i].item())
                for token in top_10_first_context:
                    word = tokenizer.decode([token])
                    new_sentence = sent.replace(tokenizer.mask_token, word)
                    print(new_sentence)
                    report.write(new_sentence + "\n")

                print("BUZZZ THIS SENTENCE IS DONE")


                #here, the score metric u r creating will go here, using the all_predictions dictionary and the 5 words for each 
                #context i've written

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

                print(all_predictions["tire"])

                
report.close()
