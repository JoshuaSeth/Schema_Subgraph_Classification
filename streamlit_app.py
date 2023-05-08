'''Streamlit app to visualize the result of applying triple extraction to future research sentences or directions. The results are pre-parsed and pickled and not parsed on the fly.'''


from datasets import load_dataset
import pickle
from annotated_text import annotated_text
import streamlit as st
import subprocess
from spacy import displacy
from st_helpers import merge_words_and_entities, visualise_dsouza_parser, visualize_dygie_parser, visualize_mechanic_granular_parser

# Default interface options
st.header('NER & RE Parsing Results')

st.markdown(
    'The analysis of these results can be found in the [research notes](https://docs.google.com/document/d/1i5xHfUvWKcGeX7D1r3Eb1IPm4Bg83-Y0/edit#bookmark=id.jb6w6xm4vqf2).')

option = st.selectbox(
    'Joint NER and RE method',
    ("D'Souza's CL-TitleParser", 'Dygie SciErc', 'Dygie GENIA', 'Dygie Ace05_Rels', 'Dygie Ace05_Event', 'Dygie MECHANIC-coarse', 'Dygie MECHANIC-granular'))

use_both = st.checkbox(
    'Use challenges OR directions vs. use challenges AND directions')

st.divider()

# ----- ----- ----- ----- ----- ----- ----- ---- ----

# Check which option was selected and visualize the parsed sentences form the corresponding schema
if option == "D'Souza's CL-TitleParser":
    visualise_dsouza_parser()


if "Dygie" in option:
    # Some preprocessing on model name
    model = option.replace('MECHANIC-', '').replace('05',
                                                    '').split()[1].lower()

    # Data is preparsed and pickled and loaded here (so not parsed on the fly)
    prefix = ''
    if use_both:
        prefix = 'pred_'
    with open(f'ORKG_parsers/dygiepp/predictions/{prefix}{model}.jsonl', 'r') as f:
        data = f.read()
    data = eval(data)

    # Choose appropriate visualizer
    if not 'granular' in option.lower():
        visualize_dygie_parser(data)

    if 'granular' in option.lower():
        visualize_mechanic_granular_parser(data)
