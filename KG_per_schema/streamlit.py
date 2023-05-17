'''Interface for the KG_per_schema module.'''
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import glob
import subprocess
from utils import project_path, get_model_fname
from results_loader import load_data
from annotated_text import annotated_text

# Options
schemas = ['scierc', 'None', 'genia', 'covid-event', 'ace05', 'ace-event']
modes = ['AND', 'OR']

# Default interface options
st.header('NER & RE Parsing Results')

st.markdown(
    'The analysis of these results can be found in the [research notes](https://docs.google.com/document/d/1i5xHfUvWKcGeX7D1r3Eb1IPm4Bg83-Y0/edit#bookmark=id.jb6w6xm4vqf2).')

schema = st.selectbox('Schema', schemas)
mode = st.selectbox('Mode', modes)
use_context = st.checkbox('Use context', value=True)
st.divider()


if schema != None and mode != None:
    sents, corefs, rels, ents = load_data(schema, mode, use_context)

    for ent in ents:
        annotated_text(ent)
