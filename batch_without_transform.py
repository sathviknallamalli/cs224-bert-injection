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
input_sentences = ["The man drove the car with a broken " + tokenizer.mask_token + " to the mechanic", "The man drove the car with a broken " + tokenizer.mask_token + " to the mechanic", 
                   "the landlord painted all the walls with " + tokenizer.mask_token + " before anyone saw", "the landlord painted all the walls with " + tokenizer.mask_token + " before anyone saw", 
        "The doctor examined the patient with a " + tokenizer.mask_token + " but could not determine the problem", "The doctor examined the patient with a " + tokenizer.mask_token + " but could not determine the problem", 
        "They finally decided to read the books on the " + tokenizer.mask_token + " so that they would not fail their history test",  "They finally decided to read the books on the " + tokenizer.mask_token + " so that they would not fail their history test",
        "the cops scared the public with " + tokenizer.mask_token + " during the parade", "the cops scared the public with " + tokenizer.mask_token + " during the parade",
        "The band played music for animals on the " + tokenizer.mask_token + " last week", "The band played music for animals on the " + tokenizer.mask_token + " last week",
        "The athlete trained before the dinner in the " + tokenizer.mask_token + " so he can feel good", "The athlete trained before the dinner in the " + tokenizer.mask_token + " so he can feel good"]

for sentenceIdx in range(0, len(input_sentences)):
    print("sentence: ", input_sentences[sentenceIdx])

    # obtaining distance matrix 
    with open('data/distance_finals.pkl', 'rb') as f:
        distance_matrices = pickle.load(f)

    # tokenizing the input sentence
    input = tokenizer(input_sentences[sentenceIdx], return_tensors = "pt")

    print(input.word_ids())

    # Run the input through the submodel to get the hidden states
    output = submodel(**input)

    # identifying location of the masked token
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

    # last layer of hidden states in the sub model (at layer 7)
    hidden_states = output.hidden_states[-1] 

    # setting learning rate & multipling hidden states by b_matrix
    lr = 0.001
    transformed_hidden = hidden_states[0]
    # splitting each word's hidden state into a separate vector
    """ split_hvecs = [transformed_hidden[i].requires_grad_(True) for i in range(0, transformed_hidden.size()[0])]
    for hvec in split_hvecs:
        hvec.retain_grad() """

    # defining a baseline loss function
    def custom_loss(matrix_1, matrix_2):
        loss = torch.mean(torch.square(matrix_1 - matrix_2))
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
    hidden_square = transformed_hidden_no_padding_first.unsqueeze(1).expand(transformed_hidden_no_padding_first.size()[0], 
                                                                    transformed_hidden_no_padding_first.size()[0], 
                                                                    transformed_hidden_no_padding_first.size()[1])
    diffs = torch.linalg.norm(torch.transpose(hidden_square, 0, 1) - hidden_square, ord = 2, dim = 2)**2
    diffs.requires_grad_(True)

    initialloss = custom_loss(diffs, distance_first_context)
    

    # training loop for first linguistic context, 10 epochs
    print('Training for first linguistic context')
    difftemp = 0
    loss = 100000
    i = 0
    convergence_threshold = 0.009
    maxiters = 10000000
    print('Initial loss: ', initialloss)

    while loss > initialloss * 0.025:
        i += 1
        # computing pairwise distances between every pair of hidden states in a sequence
        hidden_square = transformed_hidden_no_padding_first.unsqueeze(1).expand(transformed_hidden_no_padding_first.size()[0], 
                                                                        transformed_hidden_no_padding_first.size()[0], 
                                                                        transformed_hidden_no_padding_first.size()[1])
        diffs = torch.linalg.norm(torch.transpose(hidden_square, 0, 1) - hidden_square, ord = 2, dim = 2)**2

        # computing loss between the computed pariwise distances and the distance matrix for the first linguistic context
        # we can do this because hidden_square is now in the first linguistic context
        loss = custom_loss(diffs, distance_first_context)

        loss.backward(retain_graph=True)
        if i % 100 == 0:
            print("Loss at step", i, "is", loss.item())
        
        transformed_hidden_no_padding_first.retain_grad()
        transformed_hidden_no_padding_first -= lr*transformed_hidden_no_padding_first.grad.data

        transformed_hidden_no_padding_first.grad.data.zero_() 

    print("convergence loss at step", i, "is", loss.item())
    new_hidden_first = torch.cat((transformed_hidden[0].unsqueeze(0), transformed_hidden_no_padding_first, transformed_hidden[-1].unsqueeze(0)), 0)

    #untransform
    updated_hidden_first = new_hidden_first
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
    with open("data4/dumpfile" + str(sentenceIdx), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([transformed_hidden_no_padding_first, hidden_square, distance_first_context,
                    new_hidden_first, updated_hidden_first, 
                    output_first_context, logits_first_context, softmax_first_context,
                    mask_word_first_context, top_10_prob_first_context, top_10_first_context], f)
    print("BUZZZ THIS SENTENCE IS DONE")


