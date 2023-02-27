import copy
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from torch.nn import functional as F
import torch.nn as nn

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')

# Create a submodel using the first 5 layers of BERT
submodel = BertForMaskedLM.from_pretrained('bert-base-cased', output_hidden_states=True, return_dict = True, num_hidden_layers=5)
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



hidden_states = output.hidden_states[-1]

print("before split")
print(hidden_states)

#PERFORM THE MODIFICATION HERE
#apply a modification to the hidden states by adding some random value
hidden_states = hidden_states + torch.randn(hidden_states.shape)

oldModuleList = model.bert.encoder.layer
newModuleList = nn.ModuleList()

# Now iterate over all layers, only keepign only the relevant layers.
for i in range(5, 12):
    newModuleList.append(oldModuleList[i])

# create a copy of the model, modify it with the new list, and return
copyOfModel = copy.deepcopy(model)
copyOfModel.bert.encoder.layer = newModuleList

#run the output of the 5 submodel with this other model
#target_bert_model(input_ids, hidden_states=hidden_states)

output2 = copyOfModel(inputs_embeds=hidden_states, return_dict = True, output_hidden_states=True)
print("after split tensor and top words")
print(output2.hidden_states[-1])

logits = output2.logits
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