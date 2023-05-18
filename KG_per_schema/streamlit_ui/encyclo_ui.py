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
    ents, rels, sents_for_ents, rels_dict = build_encyclo_data(
        schema, mode, use_context)

    # Select entities or relations
    cur_selection = st.selectbox('Entities or relations', [
        'entities', 'relations'], key='search_1')

    # Entities tab
    if cur_selection == 'entities':
        if st.session_state['current_ent'] == None:
            viz_list_all_entities(_set_cur, ents)
        else:
            viz_current_entity(_set_cur, rels, sents_for_ents)

    # Relations tab
    if cur_selection == 'relations':
        if st.session_state['current_rel'] == None:
            viz_list_all_relations(_set_cur, rels_dict)
        else:
            viz_current_relation(_set_cur, rels_dict)


@st.cache_data(persist="disk")
def build_encyclo_data(schema, mode, use_context):
    # Load the data
    sents, corefs, rels,  entitity_sents,  = load_data(
        schema, mode, use_context, grouped=False)

    # Flatten relations
    rels = [rel for sent in rels for rel in sent]

    # Append the entity types to the relations of the graph
    for ent_sent in entitity_sents:
        for part in ent_sent:
            if isinstance(part, tuple):
                rels.append((part[0], part[1], 'is a'))

    # Create a dict with rel name: rel
    rels_dict = defaultdict(list)
    for rel in rels:
        rels_dict[rel[2]].append(rel)

    # Create a dict with ent name: sentences wherein it appears
    sents_for_ents = defaultdict(list)
    for s in entitity_sents:
        for part in s:
            if isinstance(part, tuple):
                sents_for_ents[part[0]].append(s)

    # Extract the acgual entities from the entity sentences
    ents = [
        part[0] for sent in entitity_sents for part in sent if isinstance(part, tuple)]

    # Create a dict with ent name: relations with this ent
    ents = {ent: [rel for rel in rels if ent in rel] for ent in ents}

    return ents, rels, sents_for_ents, rels_dict


def viz_current_relation(_set_cur, rels_dict):
    '''Visualizes the current relation and all relations of this relation type. '''
    col1, col2 = st.columns([6, 1])
    col2.button(label='Back', on_click=_set_cur,
                args=(None, None), type='primary')
    col1.subheader(st.session_state['current_rel'])

    for rel in rels_dict[st.session_state['current_rel']]:
        col1, col2, col3 = st.columns(3)
        col1.button(label=rel[0], key=str(
                    uuid.uuid4()), on_click=_set_cur, args=(rel[0],))
        col2.button(label=rel[2], key=str(
                    uuid.uuid4()), on_click=_set_cur, args=(None, rel[2]))
        col3.button(label=rel[1], key=str(
                    uuid.uuid4()), on_click=_set_cur, args=(rel[1],))


def viz_list_all_relations(_set_cur, rels_dict):
    '''Visualizes all relations of the graph.'''
    for rel_name, rel in rels_dict.items():
        st.button(label=rel_name + ' (' + str(len(rel)) + ' entities) ',
                  on_click=_set_cur, args=(None, rel_name,), key=str(uuid.uuid4()))


def viz_list_all_entities(_set_cur, ents):
    '''Visualizes all entities of the graph.'''
    for ent, val in ents.items():
        st.button(label=ent + ' (' + str(len(val)) + ' relations) ',
                  on_click=_set_cur, args=(ent,))


def viz_current_entity(_set_cur, rels, sents_for_ents):
    '''Visualizes current selected entity. 1. Shows sentences in which it appears. 2. Shows relations with this entity. 3. Shows other entities in the same sentence.'''
    col1, col2 = st.columns([6, 1])

    col2.button(label='Back', on_click=_set_cur,
                args=(None,), type='primary')
    col1.subheader(st.session_state['current_ent'])

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

    # When clicking on an entity we get a list of all relations that entity is involved in
