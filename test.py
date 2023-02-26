import torch
from transformers import BertTokenizer, BertModel
from torch.nn import functional as F

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

# Create a submodel using the first 5 layers of BERT
submodel = BertModel.from_pretrained('bert-base-cased', num_hidden_layers=5)
print(submodel.encoder)

# Define the input sentence with a [MASK] token
input_sentence = "The cops shot the students with " + tokenizer.mask_token + " during the daytime in Olympia near the pond."

# Tokenize the input sentence
tokenized_text = tokenizer.tokenize(input_sentence)

# Convert the tokenized text to IDs and add the special [CLS] and [SEP] tokens
input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenized_text + ['[SEP]'])

# Create a tensor from the input IDs
input_ids_tensor = torch.tensor([input_ids])

# Get the mask position (in this case, it's the second token)
mask_position = input_ids.index(tokenizer.mask_token_id)

input = tokenizer.encode_plus(input_sentence, return_tensors = "pt")

mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

# Run the input through the submodel to get the hidden states
with torch.no_grad():
    logits = submodel(input_ids_tensor)[0]
    print(submodel(input_ids_tensor).last_hidden_state)

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
