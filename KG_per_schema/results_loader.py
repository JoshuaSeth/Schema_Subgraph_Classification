'''Loads the results from the predictions repo. Gives back entities and relations. The dygie results might be in non-unfiform dictionaries. Which is the reason for this interface.'''
import subprocess
import glob
import os
from utils import project_path
import json
from tqdm import tqdm
from copy import deepcopy
from typing import List
import pickle
from collections import defaultdict
from streamlit_agraph import Node, Edge, Config
import streamlit as st
from stqdm import stqdm

# Some variables for the operation
dygie_prediction_dir_path = project_path + '/KG_per_schema/data/predictions/'
group_info_fpath = project_path + '/KG_per_schema/data/group_info/group_info.pkl'


@st.cache_data(persist="disk")
def load_data(schema: str, mode: str = 'AND', context: bool = False, index=None, grouped=True):
    '''Loads all predicted data for certain request parameters.

    Parameters
    ------------
        schema: str
            Which schema to use. One of: [scierc, None (= mechanic coarse), genia, covid-event (= mechanic granular), ace05, ace-event]
        mode: str, Optional
            Whether to use sentences that are a research challenge or direction or are both. One of: [AND, OR]. Default: AND
        context: bool, Optional
            Whether to include context sentences or not. Default: False
        index: int, Optional
            Whether to use a specific index file. If None then all data conforming to the request params is used. If given an index only a single datafile containing the request params and this specific index is used. Which might be handy for taking small samples. Default: None
        grouped: bool, Optional
            Whether to group the data by the 'group info' group_info.pkl file. The groups result from what sentence where grouped together in a context. Only relevant when using context = true. Default: True

    Return
    -----------
        If grouped: groups: dict[group_number, list[tuples]]
            Returns a dictionary where the key is group number and the values are lists of tuples. Each tuple contains the sentences, coreferences, relations and entities respectively. These four lists contain a list for every sentence.
        If not grouped: sents, corefs, rels and ents 
            Returns four lists containing the sentences, coreferences, relations and entities respectively. These four lists contain a list for every sentence or ner/re span.
        '''

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

        rels.extend(extract_relations(data))

    # If not grouped everything if done
    if not grouped:
        return sents, corefs, rels, ents

    # Else start grouping
    # Ugliest part of the code: Since spacy has in a small number of cases incorrectly split the sentences, their groups can not be retrieved anymore from the grouping info
    with open(group_info_fpath, 'rb') as f:
        group_info = pickle.load(f)
    groups = defaultdict(list)
    found_groups = 1.0
    total_groups = 1.0
    for sent, ent, rel in zip(sents, ents, rels):
        sent_text = ' '.join(sent).replace(' ,', ',').replace(' .', '.') + ' '
        group = -1
        total_groups += 1.0
        if 'sent_text' in group_info:
            group = group_info[sent_text]
            found_groups += 1.0
        groups[group].append((sent, ent, rel))

    # Requires python>=3.7
    return dict(sorted(groups.items()))


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


def extract_relations(data: dict) -> List[List]:
    '''Extracts the relations from the data if relations in the data. 

    Parameters
    ------------
        data: dict
            An dygie prediction loaded json file as dict.

    Return
    -----------
        rels 
            List of lists of lists. Each list corresponds to a sentence. The sentence consists of multiple lists. These lists are the origin text, target text and relation tag.'''
    rels = []
    if 'predicted_relations' in data:

        flattened_sents = [word for sentence in data['sentences']
                           for word in sentence]

        for sent in data['predicted_relations']:
            rels_in_sent = []
            # Sometimes there will be another layer of nested lists and sometimes just the relation
            for sub_element in sent:
                # Sub_element is a list of relations
                if isinstance(sub_element[0], list):
                    for rel in sub_element:
                        rel_in_sent = extract_rel_items(flattened_sents, rel)
                        rels_in_sent.append(rel_in_sent)
                # Sub_element is an relation
                else:
                    rel_in_sent = extract_rel_items(
                        flattened_sents, sub_element)
                    rels_in_sent.append(rel_in_sent)

            rels.append(rels_in_sent)

    return rels


def extract_rel_items(flattened_sents: List[str], rel: list) -> tuple:
    '''Takes a single list representing a relation and returns a list of the origin text, target text and relation tag.'''
    origin_start_idx = rel[0]
    origin_end_idx = rel[1]+1
    target_start_idx = rel[2]
    target_end_idx = rel[3]+1
    rel_tag = rel[4]
    rel_in_sent = tuple([' '.join(flattened_sents[origin_start_idx:origin_end_idx]),
                         ' '.join(flattened_sents[target_start_idx:target_end_idx]), rel_tag])

    return rel_in_sent


def extract_entities(data: dict) -> List[list]:
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


def build_tagged_sent(sents: List[list], ent_list: List[list]) -> List[list]:
    '''Returns the list with the tagged sentences based on the sentences and the entities from the dygie data.

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
                tagged_sent.append(word + ' ')
                last_tag = None
                # If is entity add new tag span or extend previous span
            else:
                if last_tag == tags_idxs[global_idx]:
                    tagged_sent[-1] = (tagged_sent[-1][0] +
                                       ' ' + word, tagged_sent[-1][1])
                else:
                    tagged_sent.append(tuple([word, tags_idxs[global_idx]]))
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
            matching_fpaths.append(dygie_data_fpath)
    return matching_fpaths
