'''Retrieves the unique relationships, inclusive their amount from the SciErc dataset.
Note: This requires the "processed_data" folder from the SciErc dataset (http://nlp.cs.washington.edu/sciIE/) to be present in the repository. This folder is ignored in the gitignore due to the size.
The Research notes on the drive document the reasons and usage of the SciErc dataset.'''
import json

# It is not json, but jsonlines, so read the file per line
with open('SciErc_data/json/dev.json', 'r') as f:
    data = [json.loads(line) for line in f.readlines()]


# 1. Collect unique relations
all_rels = set()

# Data is now a list of dicts that denote an annotated abstract
for abstract in data:
    # Multiple rels per sentence, last item of the rel denotes the actual relation (first denote indicies of words)
    for rels_per_sentence in abstract['relations']: 
        for rel in rels_per_sentence:
            all_rels.add(rel[-1])

print('All unique relations:', all_rels)

# 2. Collect unique entities
all_ents = set()
for abstract in data:
    # Multiple rels per sentence, last item of the rel denotes the actual relation (first denote indicies of words)
    for ents_per_sentence in abstract['ner']: 
        for ent in ents_per_sentence:
            all_ents.add(ent[-1])

print('All unique entities:', all_ents)