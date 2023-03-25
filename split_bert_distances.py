import copy
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from torch.nn import functional as F
import torch.nn as nn

#THIS CODE FILE WAS USED TO CALCULATE THE DISTANCES BETWEEN THE TRANSFORMED HIDDEN STATES OF BERT AND ITS OUTPUT STATES
#TO SEE WHICH LAYER WOULD BE BEST FOR SPLIT - THIS WAS NOT USED AFTER THE MILESTONE AFTER REALIZING THE PROBE 
#MATRIX ONLY WORKS ON LAYER 7

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')

# Create a submodel using the first 5 layers of BERT

model = BertForMaskedLM.from_pretrained('bert-base-cased',    return_dict = True, output_hidden_states=True)
text = "The cops shot the students with " + tokenizer.mask_token + " during the daytime in Olympia near the pond."

input = tokenizer.encode_plus(text, return_tensors = "pt")
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
output = model(**input)
final_hidden_state = output.hidden_states[-1]
distances = []

for i in range(0, 12):
    submodel = BertForMaskedLM.from_pretrained('bert-base-cased', output_hidden_states=True, return_dict = True, num_hidden_layers=i)
    # Define the input sentence with a [MASK] token
    input_sentence = "The cops shot the students with " + tokenizer.mask_token + " during the daytime in Olympia near the pond."

    input = tokenizer.encode_plus(input_sentence, return_tensors = "pt")

    # Run the input through the submodel to get the hidden states
    output = submodel(**input)

    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

    logits = output.logits
    softmax = F.softmax(logits, dim = -1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    top_10_prob = torch.topk(mask_word, 10, dim = 1)[0][0]

    hidden_states = output.hidden_states[-1]
    hidden_states *= 2

    print("at the split")
    
    oldModuleList = model.bert.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for j in range(12-i, 12):
        newModuleList.append(oldModuleList[j])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.bert.encoder.layer = newModuleList

    #run the output of the 5 submodel with this other model
    #target_bert_model(input_ids, hidden_states=hidden_states)

    output2 = copyOfModel(inputs_embeds=hidden_states, return_dict = True, output_hidden_states=True)
   
    #compute difference between each vector in the first layer and the last layer
    diff = final_hidden_state - output2.hidden_states[-1]
    #calculat ethe norm of each vector in diff
    form = torch.norm(diff, dim = 2)
    #sum all the norms to get a scalar
    form = torch.sum(form)
    distances.append(form)
    #print(distances)

print(distances)
