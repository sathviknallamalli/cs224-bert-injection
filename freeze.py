import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name, return_dict = True, output_hidden_states=True)

freeze_layers = 7
modification_vector = torch.rand((1, model.config.hidden_size))

for i in range(freeze_layers):
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = False

input_sentence = "The cops shot the students with " + tokenizer.mask_token + " during the daytime in Olympia near the pond."
mask_position = 7

input_ids = torch.tensor(tokenizer.encode(input_sentence, add_special_tokens=True)).unsqueeze(0)
mask_positions = torch.tensor([mask_position])

with torch.no_grad():
    outputs = model(input_ids,  attention_mask=input_ids.ne(tokenizer.pad_token_id))
    predictions = outputs[0][0, mask_position].topk(5)

#top 5 based on the current vectors
predicted_ids = predictions.indices.tolist()
predicted_words = tokenizer.convert_ids_to_tokens(predicted_ids)
print(f"Top 5 predicted words: {predicted_words}")

# Modify the hidden vectors after the first 5 layers
hidden_states = outputs.hidden_states[-1]
#print(hidden_states)
hidden_states[:, 5:] += modification_vector

# Unfreeze the previously frozen layers
for i in range(freeze_layers):
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = True

# Use the modified model to predict the top 5 probable words for the mask, using all layers
with torch.no_grad():
    outputs = model(input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id))
    predictions = outputs[0][0, mask_position].topk(5)

# Print the top 5 predicted words for the mask after modification
predicted_ids = predictions.indices.tolist()
predicted_words = tokenizer.convert_ids_to_tokens(predicted_ids)
print(f"Top 5 predicted words after modification: {predicted_words}")
