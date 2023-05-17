'''Loads the results from the predictions repo. Gives back entities and relations. The dygie results might be in non-unfiform dictionaries. Which is the reason for this interface.'''
import subprocess
import glob
import os
from utils import project_path
import json
from tqdm import tqdm
from copy import deepcopy

# Some variables for the operation
dygie_data_dir_path = project_path + '/KG_per_schema/data/dygie_data/'


def load_data(schema: str, mode: str = 'AND', context: bool = False):
    '''Loads all predicted data for certain request parameters.

    Parameters
    ------------
        schema: str
            Which schema to use. One of: [scierc, None (= mechanic granular), genia, covid-event (= mechanic coarse), ace05, ace-event]
        mode: str, Optional
            Whether to use sentences that are a research challenge or direction or are both. One of: [AND, OR]. Default: AND
        context: bool, Optional
            Whether to include context sentences or not. Default: False

    Return
    -----------
        sents, corefs, rels and ents 
            Returns four lists containing the sentences, coreferences, relations and entities respectively. These four lists contain a list for every sentence or ner/re span.'''

    # Check input
    if schema not in ["scierc", "None", "genia", "covid-event", "ace05", "ace-event"]:
        raise ValueError(
            "Schema must be one of: [scierc, None (= mechanic granular), genia, covid-event (= mechanic coarse), ace05, ace-event]")

    if mode not in ["AND", "OR"]:
        raise ValueError("Mode must be one of: [AND, OR]")

    # Retrieve file paths that match the request
    matching_fpaths = []
    for dygie_data_fpath in tqdm(glob.glob(f"{dygie_data_dir_path}*")):
        # Retrieve properties
        properties = os.path.basename(dygie_data_fpath).split('_')
        has_start_idx = properties[-1]
        has_context = 'context' in os.path.basename(dygie_data_fpath)
        has_mode = properties[2]
        has_schema = properties[0]

        # Check if the properties match the request
        if has_schema == schema and has_mode == mode and has_context == context:
            matching_fpaths.append(dygie_data_fpath)

    # Load and concatenate the data of the matching files
    sents = []
    corefs = []
    rels = []
    ents = []
    for dygie_data_fpath in matching_fpaths:
        with open(dygie_data_fpath, 'r') as f:
            data = json.load(f)
        sents.extend(data['sentences'])
        if 'predicted_clusters' in data:
            corefs.extend(data['predicted_clusters'])
        if 'predicted_relations' in data:
            rels.extend(data['predicted_relations'])
        if 'predicted_ner' in data:
            ents.extend(data['predicted_ner'])

    return sents, corefs, rels, ents


# Test
sents, corefs, rels, ents = load_data('genia', 'AND', False)
print(corefs)
