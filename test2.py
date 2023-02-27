import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizer

# Load pre-trained BERT model and tokenizer
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict = True, output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define function to inject new hidden vectors after layer 5
def inject_hidden_vectors(hidden_states, new_hidden_vectors):
    modified_hidden_states = hidden_states.clone()
    modified_hidden_states[0, 5:13, :] = new_hidden_vectors
    return modified_hidden_states

# Define function to get top k predictions for masked tokens
def get_top_k_predictions(mask_logits, k=5):
    mask_probs = nn.functional.softmax(mask_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(mask_probs, k=k, dim=-1)
    top_k_indices = top_k_indices.squeeze()
    top_k_predictions = []
    for i in range(top_k_indices.size(0)):
        predictions = []
        for j in range(top_k_indices.size(1)):
            prediction = tokenizer.decode(top_k_indices[i, j].item())
            predictions.append(prediction)
        top_k_predictions.append(predictions)
    return top_k_probs, top_k_predictions

# Example masked sentence
input_sentence = "The quick [MASK] fox jumped over the lazy [MASK]."

# Tokenize input sentence
input_tokens = tokenizer.encode(input_sentence, add_special_tokens=True, return_tensors='pt')

# Get indices of masked tokens
masked_indices = torch.where(input_tokens == tokenizer.mask_token_id)

# Pass input tokens through BERT model to get hidden states
with torch.no_grad():
    outputs = bert_model(input_tokens)
    hidden_states = outputs.hidden_states

# Get hidden states after layer 5
hidden_states_after_layer_5 = hidden_states[5]

# Double hidden vectors after layer 5
doubled_hidden_vectors = 2 * hidden_states_after_layer_5

# Inject new hidden vectors after layer 5
modified_hidden_states = inject_hidden_vectors(hidden_states, doubled_hidden_vectors)

# Pass modified hidden states through the rest of BERT to predict top 5 most probable words for masked tokens
mask_logits = bert_model(inputs_embeds=modified_hidden_states).logits
mask_logits = mask_logits[0, masked_indices[0], masked_indices[1], :]
top_k_probs, top_k_predictions = get_top_k_predictions(mask_logits, k=5)

for i, masked_index in enumerate(masked_indices[0]):
    print(f"Masked token {i+1}:")
    for j, prediction in enumerate(top_k_predictions[i]):
        print(f"  {j+1}. {prediction} ({top_k_probs[i, j].item():.4f})")
