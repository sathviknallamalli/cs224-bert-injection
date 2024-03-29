from transformers import pipeline
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased',    return_dict = True, output_hidden_states=True)
# text_sentences = ["The man drove the car with a broken " + tokenizer.mask_token + " to the mechanic",
#                   "The landlord painted all the walls with " + tokenizer.mask_token + " before anyone saw",
#                   "The doctor examined the patient with a " + tokenizer.mask_token + " but could not determine the problem",
#                   "They finally decided to read the books on the " + tokenizer.mask_token + " so that they would not fail their history test",
#                   "The cops scared the public with " + tokenizer.mask_token + " during the parade",
#                   "The band played music for animals on the " + tokenizer.mask_token + " last week",
#                   "The athlete trained before the dinner in the " + tokenizer.mask_token + " so he can feel good"]


text_sentences = ['The thieves stole all the paintings in the ' + tokenizer.mask_token + ' while the guard slept.',
 'The thieves stole all the paintings in the ' + tokenizer.mask_token + ' while the guard slept.',
 'The tourist learned the route through the ' + tokenizer.mask_token + ' while traveling on vacation.',
 'The tourist learned the route through the ' + tokenizer.mask_token + ' while traveling on vacation.',
 'The administrator announced the cuts in the ' + tokenizer.mask_token + ' even though he knew it would create hard feelings.',
 'The administrator announced the cuts in the ' + tokenizer.mask_token + ' even though he knew it would create hard feelings.',
 'The engineers designed the bridge over the ' + tokenizer.mask_token + ' but a year passed before it was built.',
 'The engineers designed the bridge over the ' + tokenizer.mask_token + ' but a year passed before it was built.',
 'The report described the government ’s programs in ' + tokenizer.mask_token + ' but most people ignored it.',
 'The report described the government ’s programs in ' + tokenizer.mask_token + ' but most people ignored it.',
 'The spy had the plans for a ' + tokenizer.mask_token + ' but he was caught before he could sell them.',
 'The spy had the plans for a ' + tokenizer.mask_token + ' but he was caught before he could sell them.',
 'The President suggested a solution to the ' + tokenizer.mask_token + ' although he knew it would be rejected.',
 'The President suggested a solution to the ' + tokenizer.mask_token + ' although he knew it would be rejected.',
 'The corporate executive considered the issues under ' + tokenizer.mask_token + ' because his career depended on the outcome.',
 'The corporate executive considered the issues under ' + tokenizer.mask_token + ' because his career depended on the outcome.',
 'The woman married the man with ' + tokenizer.mask_token + ' while her friends looked on with envy.',
 'The woman married the man with ' + tokenizer.mask_token + ' while her friends looked on with envy.',
 'The doctor cured the woman with ' + tokenizer.mask_token + ' even though his colleagues had thought it unlikely.',
 'The doctor cured the woman with ' + tokenizer.mask_token + ' even though his colleagues had thought it unlikely.',
 'The hospital admitted the patient with ' + tokenizer.mask_token + ' because she required intensive care.',
 'The hospital admitted the patient with ' + tokenizer.mask_token + ' because she required intensive care.',
 'John ordered a pizza with ' + tokenizer.mask_token + ' when he was finished studying for his calculus exam.',
 'John ordered a pizza with ' + tokenizer.mask_token + ' when he was finished studying for his calculus exam.',
 'The Vietnam veteran identified his old buddy from the ' + tokenizer.mask_token + ' even though many years had passed since he had seen him.',
 'The Vietnam veteran identified his old buddy from the ' + tokenizer.mask_token + ' even though many years had passed since he had seen him.',
 'The little girl tried to cut the apple with plastic ' + tokenizer.mask_token + ' though she was n’t very successful.',
 'The little girl tried to cut the apple with plastic ' + tokenizer.mask_token + ' though she was n’t very successful.',
 'The landlord painted all the walls with ' + tokenizer.mask_token + ' though it did n’t help the appearance of the place.',
 'The landlord painted all the walls with ' + tokenizer.mask_token + ' though it did n’t help the appearance of the place.',
 'Jane finally decided to read the books on the ' + tokenizer.mask_token + ' so that she would n’t fail her history test.',
 'Jane finally decided to read the books on the ' + tokenizer.mask_token + ' so that she would n’t fail her history test.',
 'The executive only called people on the ' + tokenizer.mask_token + ' because he was paranoid.',
 'The executive only called people on the ' + tokenizer.mask_token + ' because he was paranoid.',
 'The kids played all the albums on the ' + tokenizer.mask_token + ' before they went to bed.',
 'The kids played all the albums on the ' + tokenizer.mask_token + ' before they went to bed.',
 'That kid hit the girl with a ' + tokenizer.mask_token + ' before he got off the subway.',
 'That kid hit the girl with a ' + tokenizer.mask_token + ' before he got off the subway.',
 'The doctor examined the patient with a ' + tokenizer.mask_token + ' but he could n’t determine what the problem was.',
 'The doctor examined the patient with a ' + tokenizer.mask_token + ' but he could n’t determine what the problem was.']



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