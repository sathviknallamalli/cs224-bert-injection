import copy
import torch
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM
from torch.nn import functional as F
import torch.nn as nn
import pickle
from tqdm import tqdm
from scipy.linalg import null_space
from transformers import logging
logging.set_verbosity_error()

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')

# Define the input sentence with a [MASK] token

input_sentences = ['The thieves stole all the paintings in the ' + tokenizer.mask_token + ' while the guard slept.',
 'The thieves stole all the paintings in the ' + tokenizer.mask_token + ' while the guard slept.',
 'The tourist learned the route through the ' + tokenizer.mask_token + ' while traveling on vacation.',
 'The tourist learned the route through the ' + tokenizer.mask_token + ' while traveling on vacation.',
 'The administrator announced the cuts in the ' + tokenizer.mask_token + ' even though he knew it would create hard feelings.',
 'The administrator announced the cuts in the ' + tokenizer.mask_token + ' even though he knew it would create hard feelings.',
 'The engineers designed the bridge over the ' + tokenizer.mask_token + ' but a year passed before it was built.',
 'The engineers designed the bridge over the ' + tokenizer.mask_token + ' but a year passed before it was built.',
 'The report described the government ’s programs in ' + tokenizer.mask_token + ' but most people ignored it.',
 'The report described the government ’s programs in ' + tokenizer.mask_token + ' but most people ignored it.',
 'The spy had the plans for a ' + tokenizer.mask_token + ' but he was caught before he could sell them.',
 'The spy had the plans for a ' + tokenizer.mask_token + ' but he was caught before he could sell them.',
 'The President suggested a solution to the ' + tokenizer.mask_token + ' although he knew it would be rejected.',
 'The President suggested a solution to the ' + tokenizer.mask_token + ' although he knew it would be rejected.',
 'The corporate executive considered the issues under ' + tokenizer.mask_token + ' because his career depended on the outcome.',
 'The corporate executive considered the issues under ' + tokenizer.mask_token + ' because his career depended on the outcome.',
 'The woman married the man with ' + tokenizer.mask_token + ' while her friends looked at her yesterday.',
 'The woman married the man with ' + tokenizer.mask_token + ' while her friends looked at her yesterday.',
 'The doctor cured the woman with ' + tokenizer.mask_token + ' even though his colleagues had thought it unlikely.',
 'The doctor cured the woman with ' + tokenizer.mask_token + ' even though his colleagues had thought it unlikely.',
 'The hospital admitted the patient with ' + tokenizer.mask_token + ' because she required intensive care.',
 'The hospital admitted the patient with ' + tokenizer.mask_token + ' because she required intensive care.',
 'John ordered a pizza with ' + tokenizer.mask_token + ' when he was finished studying for his calculus exam.',
 'John ordered a pizza with ' + tokenizer.mask_token + ' when he was finished studying for his calculus exam.',
 'The Vietnam veteran identified his old buddy from the ' + tokenizer.mask_token + ' even though many years had passed since he had seen him.',
 'The Vietnam veteran identified his old buddy from the ' + tokenizer.mask_token + ' even though many years had passed since he had seen him.',
 'The little girl tried to cut the apple with plastic ' + tokenizer.mask_token + ' though she was n’t very successful.',
 'The little girl tried to cut the apple with plastic ' + tokenizer.mask_token + ' though she was n’t very successful.',
 'The landlord painted all the walls with ' + tokenizer.mask_token + ' though it did n’t help the appearance of the place.',
 'The landlord painted all the walls with ' + tokenizer.mask_token + ' though it did n’t help the appearance of the place.',
 'Jane finally decided to read the books on the ' + tokenizer.mask_token + ' so that she would n’t fail her history test.',
 'Jane finally decided to read the books on the ' + tokenizer.mask_token + ' so that she would n’t fail her history test.',
 'The executive only called people on the ' + tokenizer.mask_token + ' because he was paranoid.',
 'The executive only called people on the ' + tokenizer.mask_token + ' because he was paranoid.',
 'The kids played all the albums on the ' + tokenizer.mask_token + ' before they went to bed.',
 'The kids played all the albums on the ' + tokenizer.mask_token + ' before they went to bed.',
 'That kid hit the girl with a ' + tokenizer.mask_token + ' before he got off the subway.',
 'That kid hit the girl with a ' + tokenizer.mask_token + ' before he got off the subway.',
 'The doctor examined the patient with a ' + tokenizer.mask_token + ' but he could n’t determine what the problem was.',
 'The doctor examined the patient with a ' + tokenizer.mask_token + ' but he could n’t determine what the problem was.']

with open('/Users/sathviknallamalli/cs224-bert-injection/data/paper_new.pkl', 'rb') as f:
    distance_matrices = pickle.load(f)

distance_dict = {}
for i in range(len(input_sentences)):
    if input_sentences[i] not in distance_dict:
        distance_dict[input_sentences[i]] = []
    distance_dict[input_sentences[i]].append(distance_matrices[i])

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

#first arg - vector version of the transformed hidden
#second arg - matrix verion of the transformed hidden
#third arg - bert original vectors, we dont deviate from
#fourth arg - gold distance matrix
#fifth arg - theta weight

report = open("report_file.txt", "w")


def custom_dual_scaled_loss(hidden_vectors, hidden_matrix, ground_truth_vectors, dist_context, theta, untouched_hidden, nullspace_B, gold_A):

    #let's untransform the changed hidden vectors here
    # hidden_vectors.shape =  64, 12
    
    #untransform
    untransformed_hidden = unapply_transformation(torch.transpose(b_matrix, 0, 1), torch.transpose(hidden_vectors, 0, 1)) #768, 12
    untransformed_hidden = torch.transpose(untransformed_hidden, 0, 1) #12,768

    #untransform the untouched portion using the pinv_nullspace
    #pinvnullspace = 704, 768
    #untouched_hidden = 704, 12
    untransformed_untouched_hidden = unapply_transformation(torch.transpose(torch.tensor(nullspace_B), 0, 1), untouched_hidden) #768, 12
    untransformed_untouched_hidden = torch.transpose(untransformed_untouched_hidden, 0, 1) #12,768

    complete_untransformed = untransformed_hidden + untransformed_untouched_hidden
    #complete_untransformed = untransformed_hidden

    #loss1 = torch.linalg.norm(ground_truth_vectors - complete_untransformed)**2
    loss1 = torch.norm(torch.square(complete_untransformed - ground_truth_vectors), 2)
    
    #hidden_square_scaled = (torch.linalg.norm(dist_context)/torch.linalg.norm(hidden_matrix))*hidden_matrix
    hidden_square_scaled = hidden_matrix
    
    diffs = torch.linalg.norm(torch.transpose(hidden_square_scaled, 0, 1) - hidden_square_scaled, ord = 2, dim = 2)**2
    diffs.requires_grad_(True)
    
    #loss2 = torch.matmul(torch.from_numpy(gold_A).type(torch.DoubleTensor), torch.square(diffs - dist_context))
    loss2 = torch.mean(torch.matmul(torch.from_numpy(gold_A).type(torch.DoubleTensor), torch.square(diffs - dist_context)))
    #print("loss 1 ", loss1)
    #print('loss 2 ', loss2)
    """ report.write(f'loss 1 (from regularization): {loss1} \n \n ')
    report.write(f'\n loss 2 (from the golden distances): {loss2} \n')
    report.write("\n") """

    loss = theta * loss1 + (1-theta)*loss2
    #print('loss to gold ', loss)

    return loss

#CHOOSE THE SENTENCE HERE

from matplotlib.pylab import plt


probe_directory = 'bert-base-distance-probes/64dim-probes'

all_b_matricies = []
for i in range(4, 10):
    import_b = torch.load(probe_directory + '/BERTbase-rank64-layer'+str(i)+'.params', map_location='cpu')
    b_matrix = import_b['proj'] #768, 64
    all_b_matricies.append(b_matrix)

for layerCount in range(4,10):

        
    # Create a submodel using the first x layers of BERT
    submodel = BertForMaskedLM.from_pretrained('bert-base-cased', output_hidden_states=True, return_dict = True, num_hidden_layers = layerCount)
    print('results intervening after layer ', layerCount)

    TARGET_SENTENCE = input_sentences[22]

    distance_first_context = torch.from_numpy(distance_dict[TARGET_SENTENCE][1])
    distance_first_context = distance_first_context**2
    distance_first_context.requires_grad_(True)

    gold_1 = torch.from_numpy(distance_dict[TARGET_SENTENCE][0])
    gold_2 = torch.from_numpy(distance_dict[TARGET_SENTENCE][1])

    gold_A = np.not_equal(distance_dict[TARGET_SENTENCE][0], distance_dict[TARGET_SENTENCE][1]).astype(int)

    b_matrix = all_b_matricies[layerCount-4]

    input = tokenizer(TARGET_SENTENCE, return_tensors = "pt")
    output = submodel(**input)
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

    report.write(f"sentence {TARGET_SENTENCE}\n")
    #report.write(f"Associated Distance Matrices: {distance_dict[TARGET_SENTENCE]}\n")

    # last layer of hidden states in the sub model (at layer 7)
    hidden_states = output.hidden_states[-1] 
    compare_to_bert_space_vectors = hidden_states[0][1:-1] #12,768

    transposed_b_matrix = torch.transpose(b_matrix, 0 , 1) # 64, 768

    transformed_hidden = apply_transformation(transposed_b_matrix, torch.transpose(compare_to_bert_space_vectors, 0, 1)) #64, 12
    transformed_hidden = torch.transpose(transformed_hidden, 0, 1)
    null_space_B = null_space(transposed_b_matrix) #768, 704

    untouched_hidden = apply_transformation(torch.transpose(torch.tensor(null_space_B), 0, 1), torch.transpose(compare_to_bert_space_vectors, 0, 1)) #704, 12

    transformed_hidden.retain_grad()

    # distance matrix for first linguistic context
    

    theta = 0.1

    #transformed_hidden is the hidden vectors in the 64 dimension - RELEVANT
    #untouched_hidden is from the nullspace block and these are the things we add back when comparing loss
    #gradient updates are still being done on the transformed_hidden 64 dimension version
    
    #distance matrix version
    first_transformation = transformed_hidden.unsqueeze(1).expand(transformed_hidden.size()[0], 
                                                                    transformed_hidden.size()[0], 
                                                                    transformed_hidden.size()[1])
    
    intitialloss = custom_dual_scaled_loss(transformed_hidden, first_transformation, compare_to_bert_space_vectors, distance_first_context, theta, untouched_hidden, null_space_B, gold_A)

    report.write(f"Initial loss: {intitialloss}\n")

    itercount = 0
    lr = 0.001
    loss = intitialloss


    plt.ion()  # Enable interactive mode

    loss_values = []

    while itercount<100:
        itercount+=1
        hidden_square = transformed_hidden.unsqueeze(1).expand(transformed_hidden.size()[0], 
                                                                        transformed_hidden.size()[0], 
                                                                        transformed_hidden.size()[1])
        
        loss = custom_dual_scaled_loss(transformed_hidden, hidden_square, compare_to_bert_space_vectors, distance_first_context, theta, untouched_hidden, null_space_B, gold_A)
        #print("total loss: ", loss)
        #report.write(f"total loss: {loss}\n")
        loss_values.append(loss.item())
        """ plt.clf()
        plt.plot(range(1, itercount + 1), loss_values)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')
        plt.grid(True)
        plt.pause(0.001)  # Pause for a short duration to update the plot """


        loss.backward(retain_graph=True)
        if itercount % 100 == 0:
            print("loss at step ", itercount, " is ", loss.item())
            report.write(f'Loss at step {itercount} is {loss.item()}.\n')

        transformed_hidden.retain_grad()
        transformed_hidden -= lr * transformed_hidden.grad.data
        transformed_hidden.grad.data.zero_() 

    print("convergence loss at step", itercount, "is", loss.item())






    untransformed_hidden = unapply_transformation(torch.transpose(b_matrix, 0, 1), torch.transpose(transformed_hidden, 0, 1)) #768, 12
    untransformed_hidden = torch.transpose(untransformed_hidden, 0, 1) #12,768

    untransformed_untouched_hidden = unapply_transformation(torch.transpose(torch.tensor(null_space_B), 0, 1), untouched_hidden) #768, 12
    untransformed_untouched_hidden = torch.transpose(untransformed_untouched_hidden, 0, 1) #12,768

    complete_untransformed = untransformed_hidden + untransformed_untouched_hidden

    complete_untransformed = torch.cat((hidden_states[0][0].unsqueeze(0), complete_untransformed, hidden_states[0][-1].unsqueeze(0)), 0)
    complete_untransformed = complete_untransformed.unsqueeze(0)


    oldModuleList = model.bert.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keeping only the relevant layers.
    for i in range(layerCount, 12):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.bert.encoder.layer = newModuleList

    output_first_context = copyOfModel(inputs_embeds = complete_untransformed, return_dict = True, output_hidden_states = True)

    print('First context, top 10 words')
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
        new_sentence = TARGET_SENTENCE.replace(tokenizer.mask_token, word)
        print(new_sentence)
        report.write(new_sentence + "\n")
