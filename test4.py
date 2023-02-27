import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name, return_dict = True, output_hidden_states=True)

# Set the modification vector and the number of layers to stop at
modification_vector = torch.zeros((1, model.config.hidden_size))
stop_layer = 5

# Define the input sentence and mask position
input_sentence = "The [MASK] ate the pizza"
mask_position = 2

# Encode the input sentence with the tokenizer and convert to PyTorch tensors
input_ids = torch.tensor(tokenizer.encode(input_sentence, add_special_tokens=True)).unsqueeze(0)
mask_positions = torch.tensor([mask_position])

# Use the model to predict the top 5 probable words for the mask, using only the first 5 layers
with torch.no_grad():
    outputs = model(input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id))
    hidden_states = outputs.hidden_states[:stop_layer]
    predictions = outputs[0][0, mask_position].topk(5)

# Print the top 5 predicted words for the mask
predicted_ids = predictions.indices.tolist()
predicted_words = tokenizer.convert_ids_to_tokens(predicted_ids)
print(f"Top 5 predicted words: {predicted_words}")

# Modify the hidden vectors of the first 5 layers

# Concatenate the modified hidden states with the remaining layers and pass them through the model
with torch.no_grad():
    outputs = model(inputs_embeds=torch.cat([hidden_states, outputs[2][stop_layer:]], dim=0))

# Use the modified model to predict the top 5 probable words for the mask, using all layers
with torch.no_grad():
    predictions = outputs[0][0, mask_position].topk(5)

# Print the top 5 predicted words for the mask after modification
predicted_ids = predictions.indices.tolist()
predicted_words = tokenizer.convert_ids_to_tokens(predicted_ids)
print(f"Top 5 predicted words after modification: {predicted_words}")
