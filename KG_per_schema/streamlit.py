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
import pickle
from collections import defaultdict

# Variables
schemas = ['scierc', 'None', 'genia', 'covid-event', 'ace05', 'ace-event']
modes = ['AND', 'OR']
visualizations = ['sentences', 'graph']

# Default interface options
st.header('NER & RE Parsing Results')

st.markdown(
    'The analysis of these results can be found in the [research notes](https://docs.google.com/document/d/1i5xHfUvWKcGeX7D1r3Eb1IPm4Bg83-Y0/edit#bookmark=id.jb6w6xm4vqf2).')

schema = st.selectbox('Schema', schemas)
mode = st.selectbox('Mode', modes)
use_context = st.checkbox('Use context', value=True)
st.divider()

st.selectbox('Data', visualizations)

if schema != None and mode != None:
    groups = load_data(schema, mode, use_context, grouped=True)

    if isinstance(groups, dict):
        for idx, group in groups.items():
            st.subheader(idx)

            for sent, ent, rels in group:
                annotated_text(ent)
                for rel in rels:
                    st.text(rel)

                st.divider()
