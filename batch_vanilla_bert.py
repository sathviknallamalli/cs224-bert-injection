from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased',    return_dict = True, output_hidden_states=True)
text_sentences = ["The man drove the car with a broken " + tokenizer.mask_token + " to the mechanic",
                  "The landlord painted all the walls with " + tokenizer.mask_token + " before anyone saw",
                  "The doctor examined the patient with a " + tokenizer.mask_token + " but could not determine the problem",
                  "They finally decided to read the books on the " + tokenizer.mask_token + " so that they would not fail their history test",
                  "The cops scared the public with " + tokenizer.mask_token + " during the parade",
                  "The band played music for animals on the " + tokenizer.mask_token + " last week",
                  "The athlete trained before the dinner in the " + tokenizer.mask_token + " so he can feel good"]
for text in text_sentences:
    input = tokenizer.encode_plus(text, return_tensors = "pt")
    mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

    output = model(**input)

    logits = output.logits
    softmax = F.softmax(logits, dim = -1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]

    #get the hidden output vectors for each word in the sentence
    hidden_states = output.hidden_states

    last_hidden_states = hidden_states[-1] #this is the last layer of bert for each word 

    middle_hidden_state = hidden_states[5] #this is the last layer of bert for each word
    print("middle states")
    #print(middle_hidden_state)

    #get the last hidden state vector of each token
    print("last states")
    #print(last_hidden_states)

    #get probability of each of the top 10 words
    top_10_prob = torch.topk(mask_word, 10, dim = 1)[0][0]
    for i in range(10):
        print(tokenizer.decode([top_10[i]]), top_10_prob[i].item())
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)