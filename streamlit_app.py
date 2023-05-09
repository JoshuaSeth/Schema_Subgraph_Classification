'''Streamlit app to visualize the result of applying triple extraction to future research sentences or directions. The results are pre-parsed and pickled and not parsed on the fly.'''


from datasets import load_dataset
import pickle
from annotated_text import annotated_text
import streamlit as st
import subprocess
from spacy import displacy
from st_helpers import SchemaParser
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import numpy as np

# Helper functions


def load_sentences(use_both):
    prefix = ''
    if use_both:
        prefix = 'correct_format_'

    with open(f'ORKG_parsers/dygiepp/predictions/{prefix}scierc.jsonl', 'r') as f:
        data = f.read()
        data = eval(data)
        sentences = data['sentences']
    return prefix, sentences


def load_data_for_selected_models(options, prefix):
    model_data = {}
    for option in options:
        if 'dygie' in option.lower():
            model = option.replace('MECHANIC-', '').replace('05',
                                                            '').split()[1].lower()

        # Data is preparsed and pickled and loaded here (so not parsed on the fly)

            with open(f'ORKG_parsers/dygiepp/predictions/{prefix}{model}.jsonl', 'r') as f:
                data = f.read()
            data = eval(data)

            model_data[model] = data
    return model_data


def merge_two_merged_lists(merged_list1, merged_list2):
    result = merged_list1.copy()
    i = 0
    j = 0

    while i < len(result) and j < len(merged_list2):
        if isinstance(result[i], str) and isinstance(merged_list2[j], tuple):
            if result[i] == merged_list2[j][0]:
                result[i] = merged_list2[j]
                j += 1
        i += 1

    return result

# ----- ----- ----- ----- ----- ----- ----- ---- ----


# Default interface options
st.header('NER & RE Parsing Results')

st.markdown(
    'The analysis of these results can be found in the [research notes](https://docs.google.com/document/d/1i5xHfUvWKcGeX7D1r3Eb1IPm4Bg83-Y0/edit#bookmark=id.jb6w6xm4vqf2).')


options = st.multiselect(
    'Joint NER and RE method',
    ("D'Souza's CL-TitleParser", 'Dygie SciErc', 'Dygie GENIA', 'Dygie Ace05_Rels', 'Dygie Ace05_Event', 'Dygie MECHANIC-coarse', 'Dygie MECHANIC-granular', 'Rule-based Hearst Patterns'))


use_both = True
# st.checkbox(
# 'Use challenges OR directions vs. use challenges AND directions', True)

include_context = st.checkbox(
    'Include context: include the sentence before and afther the research sentence. \n (For some reason hidden deep in the dygie code this only works for granular MECHANIC and non-dygie models)')

compare_merge = False  # st.checkbox(
# 'Compare parsers results or merge parser results')

st.divider()

# ----- ----- ----- ----- ----- ----- ----- ---- ----

# Check which options were selected and visualize the parsed sentences form the corresponding schema
schema_parser = SchemaParser()

# Load sentences either challenging OR direction or challenge AND direction future research
_, sentences = load_sentences(use_both)

prefix = ''
if include_context:
    prefix = 'context_'

model_data = load_data_for_selected_models(options, prefix)

models_entity_counts = defaultdict(list)
models_sentence_lengths = defaultdict(list)
datas = defaultdict(lambda: defaultdict(list))

# Obtain parsed data
# For each sentence apply all selected models
for idx, s in enumerate(sentences):

    ents_for_models = []
    for option in options:
        # Load pre-parsed data
        model_name = option.replace(
            'MECHANIC-', '').replace('05', '').split()[1].lower()

        data = None
        if 'dygie' in option.lower():
            data = model_data[model_name]

        # Get idx of first word of sentence in the total text
        sent_start_idx = sum([len(sent)
                              for sent in sentences[:idx]])

        # Entities
        parsed_ents = schema_parser.parse_ents(
            s, model_name, data, idx, sent_start_idx)
        ents_for_models.append(parsed_ents)
        models_entity_counts[model_name].append(
            len([ent for ent in parsed_ents if type(ent) == tuple]))
        models_sentence_lengths[model_name].append(len(s))

        datas[model_name]['ents'].append(parsed_ents)

        # Relations
        parsed_rels = schema_parser.parse_rels(
            model_name, data, s, idx, sent_start_idx)
        datas[model_name]['rels'].append(parsed_rels)

# Note the relation between sentence length and entities
for model in models_sentence_lengths.keys():
    counts = np.array(models_entity_counts[model])
    lenghts = np.array(models_sentence_lengths[model]).reshape(-1, 1)
    reg = LinearRegression().fit(lenghts, counts)
    # print(reg.score(lenghts, counts))

    st.text(model + ' has regression coefficent ' + str(reg.coef_) +
            ' between sentence length and number of found entities')

# Visualize parsed data
for idx, s in enumerate(sentences):
    # Layout
    st.markdown('\n')
    st.subheader(str(idx))

    for option in options:

        model_name = option.replace(
            'MECHANIC-', '').replace('05', '').split()[1].lower()
        st.markdown('\n')
        st.caption(option)

        annotated_text(datas[model_name]['ents'][idx])

        parsed_rels = datas[model_name]['rels'][idx]
        for rel in parsed_rels:
            st.text(rel)
