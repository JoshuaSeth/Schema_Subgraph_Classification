import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import glob
import subprocess
from utils import project_path, get_model_fname
from results_loader import load_data, build_graph
from metrics import get_metrics, get_abs_recall_dist, get_degrees_dist, to_long_format_df
from annotated_text import annotated_text
import pickle
from collections import defaultdict
from streamlit_agraph import agraph, Config, Node, Edge
import plotly.express as px
import plotly
import uuid


@st.cache_data(persist="disk", experimental_allow_widgets=True)
def viz_encyclo_ui(schema, mode, use_context, _set_cur):
    '''Visualizes an interactive encyclopedia of entities and relations of the graph for the current selected parameters'''
    # The entry point is a list of all entities
    sents, corefs, rels,  ents,  = load_data(
        schema, mode, use_context, grouped=False)

    rels = [rel for sent in rels for rel in sent]

    rels_dict = defaultdict(list)
    for rel in rels:
        rels_dict[rel[2]].append(rel)

    sents_for_ents = defaultdict(list)

    for s in ents:
        for part in s:
            if isinstance(part, tuple):
                sents_for_ents[part[0]].append(s)

    ents = [
        part[0] for sent in ents for part in sent if isinstance(part, tuple)]

    ents = {ent: [rel for rel in rels if ent in rel] for ent in ents}

    cur_selection = st.selectbox('Entities or relations', [
        'entities', 'relations'], key='search_1')

    # Entities tab
    if cur_selection == 'entities':
        print(st.session_state['current_ent'])
        # Main scrolling menu
        if st.session_state['current_ent'] == None:
            for ent, val in ents.items():
                st.button(label=ent + ' (' + str(len(val)) + ' relations) ',
                          on_click=_set_cur, args=(ent,))
        # Visualization  for specifc item
        else:
            st.button(label='Back', on_click=_set_cur,
                      args=(None,), type='primary')
            st.subheader(st.session_state['current_ent'])

            # Viz the sentence involving this one
            st.caption('Sentences with this entity')
            for s in sents_for_ents[st.session_state['current_ent']]:
                annotated_text(s)

            st.divider()
            # Get all relations that involve the current entity
            rels_ = [
                rel for rel in rels if st.session_state['current_ent'] in rel]

            st.caption('Relations with this entity')
            for rel in rels_:
                col1, col2, col3 = st.columns(3)
                col1.button(label=rel[0], key=str(
                    uuid.uuid4()), on_click=_set_cur, args=(rel[0],))
                col2.button(label=rel[2], key=str(
                    uuid.uuid4()), on_click=_set_cur, args=(None, rel[2]))
                col3.button(label=rel[1], key=str(
                    uuid.uuid4()), on_click=_set_cur, args=(rel[1],))

            st.divider()
            st.caption('Other entities in this sentence')
            temp = [part[0]
                    for s in sents_for_ents[st.session_state['current_ent']] for part in s if isinstance(part, tuple)]
            temp = {ent: [rel for rel in rels if ent in rel]
                    for ent in temp}

            for ent, val in temp.items():
                st.button(label=ent + ' (' + str(len(val)) + ' relations) ',
                          on_click=_set_cur, args=(ent,))

    # Relations tab
    if cur_selection == 'relations':
        if st.session_state['current_rel'] == None:
            for rel_name, rel in rels_dict.items():
                st.button(label=rel_name + ' (' + str(len(rel)) + ' entities) ',
                          on_click=_set_cur, args=(None, rel_name,), key=str(uuid.uuid4()))
        else:
            st.button(label='Back', on_click=_set_cur,
                      args=(None, None), type='primary')
            st.subheader(st.session_state['current_rel'])

            for rel in rels_dict[st.session_state['current_rel']]:
                col1, col2, col3 = st.columns(3)
                col1.button(label=rel[0], key=str(
                    uuid.uuid4()), on_click=_set_cur, args=(rel[0],))
                col2.button(label=rel[2], key=str(
                    uuid.uuid4()), on_click=_set_cur, args=(None, rel[2]))
                col3.button(label=rel[1], key=str(
                    uuid.uuid4()), on_click=_set_cur, args=(rel[1],))

    # When clicking on an entity we get a list of all relations that entity is involved in
