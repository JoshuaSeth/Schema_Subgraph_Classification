'''Streamlit app to visualize the result of applying triple extraction to future research sentences or directions. The results are pre-parsed and pickled and not parsed on the fly.'''


from datasets import load_dataset
import pickle
from annotated_text import annotated_text
import streamlit as st
import subprocess
from spacy import displacy
from st_helpers import merge_words_and_entities, visualise_dsouza_parser, visualize_dygie_parser, visualize_mechanic_granular_parser, SchemaParser

# Default interface options
st.header('NER & RE Parsing Results')

st.markdown(
    'The analysis of these results can be found in the [research notes](https://docs.google.com/document/d/1i5xHfUvWKcGeX7D1r3Eb1IPm4Bg83-Y0/edit#bookmark=id.jb6w6xm4vqf2).')


options = st.multiselect(
    'Joint NER and RE method',
    ("D'Souza's CL-TitleParser", 'Dygie SciErc', 'Dygie GENIA', 'Dygie Ace05_Rels', 'Dygie Ace05_Event', 'Dygie MECHANIC-coarse', 'Dygie MECHANIC-granular'))


use_both = st.checkbox(
    'Use challenges OR directions vs. use challenges AND directions')

st.divider()

# ----- ----- ----- ----- ----- ----- ----- ---- ----

# Check which option was selected and visualize the parsed sentences form the corresponding schema


schema_parser = SchemaParser()
prefix = ''
if use_both:
    prefix = 'pred_'

with open(f'ORKG_parsers/dygiepp/predictions/{prefix}scierc.jsonl', 'r') as f:
    data = f.read()
    data = eval(data)
    sentences = data['sentences']

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


for idx, s in enumerate(sentences):
    for option in options:
        model = option.replace('MECHANIC-', '').replace('05',
                                                        '').split()[1].lower()

        print(model)

        data = model_data[model] if not model == 'cl-titleparser' else None

        st.markdown('\n')
        st.subheader(str(idx))
        sent_start_idx = sum([len(sent)
                              for sent in sentences[:idx]])

        parsed_ents = schema_parser.parse_ents(
            s, model, data, idx, sent_start_idx)
        annotated_text(parsed_ents)

        parsed_rels = schema_parser.parse_rels(
            model, data, s, idx, sent_start_idx)
        for rel in parsed_rels:
            st.text(rel)
