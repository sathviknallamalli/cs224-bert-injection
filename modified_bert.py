import torch
from transformers import BertTokenizer, BertModel

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

# Create a submodel using the first 5 layers of BERT
submodel = BertModel.from_pretrained('bert-base-cased', num_hidden_layers=5)

# Set the model to evaluation mode
submodel.eval()

# Define the input sentence with a [MASK] token
input_sentence = "The [MASK] jumped over the lazy dog."

# Tokenize the input sentence
tokenized_text = tokenizer.tokenize(input_sentence)

# Convert the tokenized text to IDs and add the special [CLS] and [SEP] tokens
input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenized_text + ['[SEP]'])

# Create a tensor from the input IDs
input_ids_tensor = torch.tensor([input_ids])

# Get the mask position (in this case, it's the second token)
mask_position = input_ids.index(tokenizer.mask_token_id)

# Run the input through the submodel to get the hidden states
with torch.no_grad():
    hidden_states = submodel(input_ids_tensor)[0]

# Get the probabilities for the top 5 probable words for the mask
mask_hidden_state = hidden_states[0][mask_position]
top_k_indices = torch.topk(mask_hidden_state, k=5).indices.tolist()
top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)

# Print the top 5 probable words for the mask
print("Top 5 probable words for the mask:")
for token in top_k_tokens:
    print(token)
