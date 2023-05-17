'''Loads the results from the predictions repo. Gives back entities and relations. The dygie results might be in non-unfiform dictionaries. Which is the reason for this interface.'''
import subprocess
import glob
import os
from utils import project_path
import json
from tqdm import tqdm
from copy import deepcopy

# Some variables for the operation
dygie_prediction_dir_path = project_path + '/KG_per_schema/data/predictions/'


def load_data(schema: str, mode: str = 'AND', context: bool = False, index=None):
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
    matching_fpaths = get_fpaths_for_request(schema, mode, context, index)

    # Load and concatenate the data of the matching files
    sents, corefs, rels, ents = [], [], [], []

    for dygie_data_fpath in matching_fpaths:
        with open(dygie_data_fpath, 'r') as f:
            data = json.load(f)

        # Extend the collected data
        sents.extend(data['sentences'])

        ents.extend(extract_entities(data))

    return sents, corefs, rels, ents


def get_tag_idxs(entity_list):
    '''Creates a dictionary of word indices and their corresponding tags.'''
    tags_idxs = {}
    for sent in entity_list:
        # Sometimes there will be another layer of nested lists and sometimes just the entity
        for sub_element in sent:
            # Sub_element is a list of entities
            if isinstance(sub_element[0], list):
                for ent in sub_element:
                    set_idx_and_tag(ent, tags_idxs)
            # Sub_element is an entity
            else:
                set_idx_and_tag(sub_element, tags_idxs)
    return tags_idxs


def set_idx_and_tag(ent, tags_idxs):
    '''Get the idx and tag from the entity and add it to the tags_idxs dict.'''
    if len(ent) > 2:
        # Sometimes the second element is end idx but sometimes it is the tag itself
        if isinstance(ent[1], int):
            for i in range(ent[0], ent[1]+1):
                tags_idxs[i] = ent[2]
        else:
            tags_idxs[ent[0]] = ent[1]


def extract_entities(data: dict) -> list[list[str | tuple]]:
    '''Extracts the entities from the data if entities in the data. 

    Parameters
    ------------
        data: dict
            An dygie prediction loaded json file as dict.

    Return
    -----------
        ents 
            List of lists. Each list is a sentence. The sentence consists of plain untagged words or tuples denoting the word and its entity tag.'''
    ents = []
    if 'predicted_ner' in data:
        # Create a dict of words indices and their corresponding tags
        l = build_tagged_sent(data['sentences'], data['predicted_ner'])
        ents.extend(l)

    if 'predicted_events' in data:
        # For events we follow the same proces
        l = build_tagged_sent(data['sentences'], data['predicted_events'])
        ents.extend(l)

    return ents


def build_tagged_sent(sents, ent_list):
    '''Fills the fill list with the tagged sentences based on the sentences and the entities from the dygie data.

    Parameters
    ------------
        sents: list
            Sentences from the dygie data.
        ent_list: list[list]
            A list of a list of entities in the form: [start_idx, end_idx, tag] or [idx, tag]

    Return
    -----------
        ents 
            List of lists. Each list is a sentence. The sentence consists of plain untagged words or tuples denoting the word and its entity tag.'''
    tags_idxs = get_tag_idxs(ent_list)

    fill_list = []
    sent_total_idx = 0
    for sent in sents:
        last_tag = None
        tagged_sent = []
        fill_list.append(tagged_sent)
        for idx, word in enumerate(sent):
            # Add plain word if not tagged
            global_idx = sent_total_idx+idx
            if not global_idx in tags_idxs:
                tagged_sent.append(word)
                last_tag = None
                # If is entity add new tag span or extend previous span
            else:
                if last_tag == tags_idxs[global_idx]:
                    tagged_sent[-1][0] += ' ' + word
                else:
                    tagged_sent.append([word, tags_idxs[global_idx]])
                    # Set this tag as the last tag
                last_tag = tags_idxs[global_idx]
        sent_total_idx += len(sent)
    return fill_list


def get_fpaths_for_request(schema, mode, context, index):
    '''Returns the file paths that match the request parameters.'''
    matching_fpaths = []
    for dygie_data_fpath in tqdm(glob.glob(f"{dygie_prediction_dir_path}*")):
        # Retrieve properties
        properties = os.path.basename(dygie_data_fpath).split('_')
        has_start_idx = properties[-1]
        has_context = 'context' in os.path.basename(dygie_data_fpath)
        has_mode = properties[-2]
        has_schema = properties[0]

        # Check if the properties match the request
        if has_schema == schema and has_mode == mode and has_context == context and (index == None or has_start_idx == str(index)):
            print(dygie_data_fpath)
            matching_fpaths.append(dygie_data_fpath)
    return matching_fpaths


# Test
sents, corefs, rels, ents = load_data('ace-event', 'AND', True, index=880)
print(ents)
