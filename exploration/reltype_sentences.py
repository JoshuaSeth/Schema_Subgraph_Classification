'''Print all the sentences with the given relation type (for example, EVALUATE-FOR), also print the 2 spans for which this relation is present'''
import json

# The relation type you want the sentnces for {'EVALUATE-FOR', 'USED-FOR', 'CONJUNCTION', 'FEATURE-OF', 'HYPONYM-OF', 'PART-OF', 'COMPARE'}
REL_TYPE = 'EVALUATE-FOR'

# It is not json, but jsonlines, so read the file per line
with open('SciErc_data/json/dev.json', 'r') as f:
    data = [json.loads(line) for line in f.readlines()]

# Data is now a list of dicts that denote an annotated abstract
for abstract in data[:]:
    # Multiple rels per sentence, last item of the rel denotes the actual relation (first denote indicies of words)
    for sentence_idx,  rels_per_sentence in enumerate(abstract['relations']): 
        for rel in rels_per_sentence:
            if rel[-1] == REL_TYPE:
                text = [word for sentence in abstract['sentences'] for word in sentence]
                span_1 = text[rel[0]: rel[1]+1]
                span_2 = text[rel[2]: rel[3]+1]

                # print('\n', ' '.join(text))
                print(span_1, span_2)

