import json
with open("dataset.json") as f:
    data = json.load(f)

report = open("john_sentences.txt", "w")

for sent in data:
    sent_base = sent["sentence"]
    #sent_1 = sent_base.replace("[MASK]", sent["attachment1_words"][0])
    #sent_2 = sent_base.replace("[MASK]", sent["attachment2_words"][0])

    sent_1 = sent_base
    sent_2 = sent_base

    report.write(f'{sent_1}\n')
    report.write(f'{sent_2}\n')

report.close()