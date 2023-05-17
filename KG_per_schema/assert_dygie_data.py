import json

with open('KG_per_schema/data/dygie_data/genia_sents_AND.jsonl', 'r') as f:
    data = json.load(f)

for sent in data['sentences']:
    if len(sent) < 3:
        print(sent)
