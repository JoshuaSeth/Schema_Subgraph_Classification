'''Reads the dygie data directory (folder with all predictions of entities and relations for all the sentences). Checks whether for all schema's the same number of files and sentences was parsed. Also checks whether there are double indices in the files per schema.'''
import json
import glob
import os
from collections import defaultdict
from utils import hash_data
'''WHile this validation has shown that the files in the prediction dir are correct in number and intact, some scierc files have mangled indices. This causes the dataloader to not load them properly since the requested properties like mode now suddently are matched to the mangled indices. Properly scierc data in the reuslts is not trustworthy.
DONE: the dataloader now also robustly loads mangled indices'''


def nested_defaultdict():
    return defaultdict(lambda: defaultdict(list))


data_per_schema = nested_defaultdict()


def recursive_items(dictionary, extend_key=""):
    for key, value in dictionary.items():
        if isinstance(value, defaultdict):
            yield from recursive_items(value, extend_key=extend_key + str(key) + ' ')
        else:
            yield extend_key + ' ' + key, value


data_per_schema = nested_defaultdict()

# Traverse dir containing all prediction data
dygie_prediction_dir_path = "KG_per_schema/data/dygie_data/"
for dygie_data_fpath in glob.glob(f"{dygie_prediction_dir_path}*"):

    # Retrieve properties
    properties = os.path.basename(dygie_data_fpath).split('_')
    has_context = 'context' in os.path.basename(dygie_data_fpath)
    has_schema = properties[0]
    if not has_context:
        mode = properties[2]
        is_research = properties[3]
        idx = properties[4:]
    else:
        mode = properties[3]
        is_research = properties[4]
        idx = properties[5:]

    if has_schema == 'scierc':
        print(dygie_data_fpath)

    # Save the file idxes per config (1st level) and schema (2nd level)
    data_per_schema[mode + ' '+str(has_context) +
                    ' '+is_research][has_schema].append(idx)


# Print the number of files per schema and configuration
for key, val in recursive_items(data_per_schema):
    print(key, len(val))

# Output:
# OR True True  None 89
# OR True True  ace05 89
# OR True True  ace-event 89
# OR True True  covid-event 89
# OR True True  genia 89
# OR True True  scierc 95
# OR True False  covid-event 96
# OR True False  None 96
# OR True False  genia 96
# OR True False  ace-event 96
# OR True False  ace05 96
# OR True False  scierc 96
# AND False True  ace-event 10
# AND False True  scierc 10
# AND False True  ace05 10
# AND False True  genia 10
# AND False True  None 10
# AND False True  covid-event 10
# AND False False  covid-event 62
# AND False False  scierc 62
# AND False False  genia 62
# AND False False  None 62
# AND False False  ace-event 62
# AND False False  ace05 62
# OR False False  ace05 37
# OR False False  None 37
# OR False False  scierc 37
# OR False False  ace-event 37
# OR False False  genia 37
# OR False False  covid-event 37
# AND True True  scierc 31
# AND True True  genia 26
# AND True True  ace-event 26
# AND True True  ace05 26
# AND True True  None 26
# AND True True  covid-event 26
# OR False True  ace05 35
# OR False True  None 35
# OR False True  ace-event 35
# OR False True  genia 35
# OR False True  covid-event 35
# OR False True  scierc 35

all_rels = defaultdict(list)

dygie_prediction_dir_path = "KG_per_schema/data/predictions/"
for dygie_data_fpath in glob.glob(f"{dygie_prediction_dir_path}*"):

    # Retrieve properties
    properties = os.path.basename(dygie_data_fpath).split('_')
    has_context = 'context' in os.path.basename(dygie_data_fpath)
    has_schema = properties[0]
    if not has_context:
        mode = properties[2]
        is_research = properties[3]
        idx = properties[4:]
    else:
        mode = properties[3]
        is_research = properties[4]
        idx = properties[5:]

    if mode == 'OR':
        with open(dygie_data_fpath, 'r') as f:
            data = json.load(f)

        if 'predicted_relations' in data:
            re = data['predicted_relations']

            for rel_sent in re:
                if len(rel_sent) > 0:

                    all_rels[has_schema].append(len(rel_sent))

        if 'predicted_ner' in data:
            ner = data['predicted_ner']

            for ner_sent in ner:
                if len(ner_sent) > 0:

                    all_rels[has_schema].append(len(ner_sent))

for key, val in all_rels.items():
    print(key, len(val))
