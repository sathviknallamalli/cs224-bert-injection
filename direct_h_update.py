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

# Create a submodel using the first 7 layers of BERT
submodel = BertForMaskedLM.from_pretrained('bert-base-cased', output_hidden_states=True, return_dict = True, num_hidden_layers = 7)

# Define the input sentence with a [MASK] token
input_sentences = ["The man drove the car with a broken " + tokenizer.mask_token + " to the mechanic", "The man drove the car with a broken " + tokenizer.mask_token + " to the mechanic", 
                   "the landlord painted all the walls with " + tokenizer.mask_token + " before anyone saw", "the landlord painted all the walls with " + tokenizer.mask_token + " before anyone saw", 
        "The doctor examined the patient with a " + tokenizer.mask_token + " but could not determine the problem", "The doctor examined the patient with a " + tokenizer.mask_token + " but could not determine the problem", 
        "They finally decided to read the books on the " + tokenizer.mask_token + " so that they would not fail their history test",  "They finally decided to read the books on the " + tokenizer.mask_token + " so that they would not fail their history test",
        "the cops scared the public with " + tokenizer.mask_token + " during the parade", "the cops scared the public with " + tokenizer.mask_token + " during the parade",
        "The band played music for animals on the " + tokenizer.mask_token + " last week", "The band played music for animals on the " + tokenizer.mask_token + " last week",
        "The athlete trained before the dinner in the " + tokenizer.mask_token + " so he can feel good", "The athlete trained before the dinner in the " + tokenizer.mask_token + " so he can feel good"]

def apply_transformation(matrix, vector):
    transformed_vector = torch.mm(matrix, vector)
    return transformed_vector

def b_loss(hidden_vectors, dist_context, theta, original_hidden_vectors):
    transformed_hidden = apply_transformation(b_matrix, torch.transpose(hidden_vectors, 0, 1)) #768,12
    transformed_hidden = torch.transpose(transformed_hidden, 0, 1) #12,768
    hidden_matrix = transformed_hidden.unsqueeze(1).expand(transformed_hidden.size()[0], 
                                                                    transformed_hidden.size()[0], 
                                                                    transformed_hidden.size()[1]) #12,12,768

    hidden_square_scaled = (torch.linalg.norm(dist_context)/torch.linalg.norm(hidden_matrix))*hidden_matrix
    
    diffs = torch.linalg.norm(torch.transpose(hidden_square_scaled, 0, 1) - hidden_square_scaled, ord = 2, dim = 2)**2
    diffs.requires_grad_(True)
    loss2 = torch.mean(torch.square(diffs - dist_context))
    loss1 = torch.norm(torch.square(hidden_vectors - original_hidden_vectors), 2)

    loss = theta*loss1 + (1-theta)*loss2
    
    return loss


for sentenceIdx in range(12, 14):
    print("sentence: ", input_sentences[sentenceIdx])

    # obtaining distance matrix 
    with open('/Users/aakritilakshmanan/cs224-bert-injection/data/distance_finals.pkl', 'rb') as f:
        distance_matrices = pickle.load(f)

    input = tokenizer(input_sentences[sentenceIdx], return_tensors = "pt")
    #print(input.word_ids())
    output = submodel(**input)
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

    import_b = torch.load('bert-base-distance-probes/ptb-prd-BERTbase-rank768.params', map_location='cpu')
    b_matrix = import_b['proj']

    # last layer of hidden states in the sub model (at layer 7)
    hidden_states = output.hidden_states[-1] 
    optimized_hidden_vectors = hidden_states[0][1:-1] #12,768 
    og_hidden_vectors = optimized_hidden_vectors.clone()
    optimized_hidden_vectors = optimized_hidden_vectors.requires_grad_(True)
    optimized_hidden_vectors.retain_grad()

    # distance matrix for first linguistic context
    distance_first_context = torch.from_numpy(distance_matrices[sentenceIdx])
    distance_first_context = distance_first_context**2
    distance_first_context.requires_grad_(True)

    theta = 0.60

    initialloss =  b_loss(optimized_hidden_vectors, distance_first_context, theta, og_hidden_vectors)

    print('Initial loss: ', initialloss)

    # training loop for first linguistic context, 10 epochs
    loss = 100000000000
    i = 0

    # setting learning rate & multipling hidden states by b_matrix
    lr = 0.001

    while i < 500:
        i += 1
        # computing pairwise distances between every pair of hidden states in a sequence
        loss = b_loss(optimized_hidden_vectors, distance_first_context, theta, og_hidden_vectors)

        loss.backward(retain_graph=True)
        if i % 100 == 0:
            print("Loss at step", i, "is", loss.item())
        
        optimized_hidden_vectors.retain_grad()
        optimized_hidden_vectors -= lr*optimized_hidden_vectors.grad.data

        optimized_hidden_vectors.grad.data.zero_() 

    print("convergence loss at step", i, "is", loss.item())
    new_hidden_first = torch.cat((hidden_states[0][0].unsqueeze(0), optimized_hidden_vectors, hidden_states[0][-1].unsqueeze(0)), 0).unsqueeze(0)
    oldModuleList = model.bert.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keeping only the relevant layers.
    for i in range(7, 12):
        newModuleList.append(oldModuleList[i])

    copyOfModel = copy.deepcopy(model)
    copyOfModel.bert.encoder.layer = newModuleList

    output_first_context = copyOfModel(inputs_embeds = new_hidden_first, return_dict = True, output_hidden_states = True)
    #output_first_context = copyOfModel(inputs_embeds = hidden_states, return_dict = True, output_hidden_states = True)
    
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


