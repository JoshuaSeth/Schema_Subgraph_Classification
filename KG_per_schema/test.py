import json

with open('KG_per_schema/data/predictions/spacy-encorewebsm_sents_OR_0') as f:
    data = json.load(f)

flattened_sents = [w for s in data['sentences'] for w in s ]

for sent, ners in zip(data['sentences'], data['predicted_ner']):
    print(sent)
    for ner in ners:
        print(flattened_sents[ner[0]:ner[1]+1], ner[2])
    print('\n')