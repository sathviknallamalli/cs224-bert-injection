from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

unmasker = pipeline('fill-mask', model='bert-base-cased')
unmasker("Hello I'm a [MASK] model.")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased',    return_dict = True, output_hidden_states=True)
text = "The cops shot the students with " + tokenizer.mask_token + " during the daytime in Olympia near the pond."

input = tokenizer.encode_plus(text, return_tensors = "pt")
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
print("here")
print(input)
output = model(**input)
print(output)
logits = output.logits
softmax = F.softmax(logits, dim = -1)
mask_word = softmax[0, mask_index, :]
top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
#get the hidden output vectors for each word in the sentence
hidden_states = output.hidden_states
#get the last_hidden_states tensor
#print(len(hidden_states)) #this is 13, for each layer of bert

last_hidden_states = hidden_states[-1] #this is the last layer of bert for each word 

middle_hidden_state = hidden_states[5] #this is the last layer of bert for each word
print("middle states")
print(middle_hidden_state)

#get the last hidden state vector of each token
print("last states")
print(last_hidden_states)
#get probability of each of the top 10 words
top_10_prob = torch.topk(mask_word, 10, dim = 1)[0][0]
for i in range(10):
    print(tokenizer.decode([top_10[i]]), top_10_prob[i].item())
for token in top_10:
   word = tokenizer.decode([token])
   new_sentence = text.replace(tokenizer.mask_token, word)
   print(new_sentence)