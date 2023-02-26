import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Load the BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Modify the BERT model
class ModifiedBERT(nn.Module):
    def __init__(self, num_layers, hidden_size, num_classes):
        super().__init__()
        self.bert = bert_model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.modifier = nn.Linear(hidden_size*2, hidden_size*2)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, hidden_states=None):
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
        else:
            hidden_states = list(hidden_states)

        # Run the input through the first 5 layers of the BERT model
        hidden_states = [self.bert.embeddings(input_ids)] + list(hidden_states[:5])
        for i in range(5, self.num_layers):
            if i == 5:
                # Double the hidden vectors after the first 5 layers
                batch_size, sequence_length, _ = hidden_states[0].size()
                hidden_states[0] = hidden_states[0].reshape(batch_size*sequence_length, self.hidden_size*2)
                hidden_states[0] = self.modifier(hidden_states[0])
                hidden_states[0] = hidden_states[0].reshape(batch_size, sequence_length, self.hidden_size*2)
            hidden_states = [self.bert.encoder.layer[i](hidden_states[0], hidden_states[1])] + list(hidden_states[2:])
        # Run the final hidden states through the classification layer
        pooled_output = self.bert.pooler(hidden_states[-1])
        logits = self.classifier(pooled_output)
        return logits
# Instantiate the modified BERT model
modified_bert = ModifiedBERT(num_layers=12, hidden_size=768, num_classes=len(tokenizer))

# Set the model to evaluation mode
modified_bert.eval()

# Define a function to predict the top 5 probable words for a masked word in a given sentence
def predict_masked_word(sentence):
    # Tokenize the input sentence and mask the target word with the [MASK] token
    tokenized_input = tokenizer.tokenize(sentence)
    masked_index = tokenized_input.index('[MASK]')
    tokenized_input[masked_index] = '[MASK]'

    # Convert the tokenized input to input IDs
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenized_input)).unsqueeze(0)

    # Predict the top 5 probable words for the masked word
    with torch.no_grad():
        outputs = modified_bert(input_ids)
        predictions = nn.functional.softmax(outputs, dim=-1)
        masked_word_predictions = predictions[0, masked_index]
        top_k = torch.topk(masked_word_predictions, k=5)

        # Convert the predicted indices back to words
        predicted_words = []
        for index in top_k.indices:
            predicted_words.append(tokenizer.convert_ids_to_tokens(index.item()))

        return predicted_words

# Test the function with a sample sentence
sentence = 'The [MASK] jumped over the lazy dog.'
predicted_words = predict_masked_word(sentence)
print(predicted_words)
