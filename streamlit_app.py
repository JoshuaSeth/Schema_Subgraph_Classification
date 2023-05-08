'''Streamlit app to visualize the result of applying triple extraction to future research sentences or directions. The results are pre-parsed and pickled and not parsed on the fly.'''


from datasets import load_dataset
import pickle
from annotated_text import annotated_text
import streamlit as st
import subprocess
from spacy import displacy
from st_helpers import merge_words_and_entities, visualise_dsouza_parser, visualize_dygie_parser, visualize_mechanic_granular_parser, SchemaParser


# Helper functions


def load_sentences(use_both):
    prefix = ''
    if use_both:
        prefix = 'pred_'

    with open(f'ORKG_parsers/dygiepp/predictions/{prefix}scierc.jsonl', 'r') as f:
        data = f.read()
        data = eval(data)
        sentences = data['sentences']
    return prefix, sentences


def load_data_for_selected_models(options, prefix):
    model_data = {}
    for option in options:
        if option != "D'Souza's CL-TitleParser":
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
    ("D'Souza's CL-TitleParser", 'Dygie SciErc', 'Dygie GENIA', 'Dygie Ace05_Rels', 'Dygie Ace05_Event', 'Dygie MECHANIC-coarse', 'Dygie MECHANIC-granular'))


use_both = st.checkbox(
    'Use challenges OR directions vs. use challenges AND directions')

compare_merge = st.checkbox(
    'Compare parsers results or merge parser results')

st.divider()

# ----- ----- ----- ----- ----- ----- ----- ---- ----

# Check which option was selected and visualize the parsed sentences form the corresponding schema


schema_parser = SchemaParser()

# Load sentences either challenging OR direction or challenge AND direction future research
prefix, sentences = load_sentences(use_both)

model_data = load_data_for_selected_models(options, prefix)

# For each sentence apply all selected models
for idx, s in enumerate(sentences):
    # Layout
    st.markdown('\n')
    st.subheader(str(idx))

    ents_for_models = []
    for option in options:
        st.markdown('\n')
        st.caption(option)

        # Load pre-parsed data
        model_name = option.replace(
            'MECHANIC-', '').replace('05', '').split()[1].lower()

        data = model_data[model_name] if not model_name == 'cl-titleparser' else None

        # Get idx of first word of sentence in the total text
        sent_start_idx = sum([len(sent)
                              for sent in sentences[:idx]])

        # Entities
        parsed_ents = schema_parser.parse_ents(
            s, model_name, data, idx, sent_start_idx)
        ents_for_models.append(parsed_ents)
        annotated_text(parsed_ents)

        # Relations
        parsed_rels = schema_parser.parse_rels(
            model_name, data, s, idx, sent_start_idx)
        for rel in parsed_rels:
            st.text(rel)

    if compare_merge:
        res = merge_two_merged_lists(ents_for_models[0], ents_for_models[1])

        annotated_text(res)
