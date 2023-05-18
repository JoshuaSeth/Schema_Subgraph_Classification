'''Interface for the KG_per_schema module.'''
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

# Variables
schemas = ['scierc', 'None', 'genia', 'covid-event', 'ace05', 'ace-event']
modes = ['AND', 'OR']
visualizations = ['sentences', 'graph', 'graph stats', 'encyclopedic explorer']

# Default interface options
st.header('NER & RE Parsing Results')

st.markdown(
    'The analysis of these results can be found in the [research notes](https://docs.google.com/document/d/1i5xHfUvWKcGeX7D1r3Eb1IPm4Bg83-Y0/edit#bookmark=id.jb6w6xm4vqf2).')

schema = st.selectbox('Schema', schemas)
mode = st.selectbox('Mode', modes)
use_context = st.checkbox('Use context', value=True)
sent_tab, graph_tab, graph_stats_tab, ecyclo_tab = st.tabs(visualizations)

# Data flow state
if not 'current_ent' in st.session_state:
    st.session_state['current_ent'] = None
if not 'current_rel' in st.session_state:
    st.session_state['current_rel'] = None


def set_cur(ent=None, rel=None):
    '''Sets the current entity and relation to the given ones. If not given the entity or relation will be set to none.'''
    st.session_state['current_ent'] = ent
    st.session_state['current_rel'] = rel
    if ent == None:
        st.session_state.search_1 = 'relations'
    if rel == None:
        st.session_state.search_1 = 'entities'


if schema != None and mode != None:
    # Visualize the sentences and the tagged entities and relations
    with sent_tab:
        groups = load_data(schema, mode, use_context, grouped=True)

        if isinstance(groups, dict):
            for idx, group in groups.items():
                st.subheader(idx)

                for sent, ent, rels in group:
                    annotated_text(ent)
                    for rel in rels:
                        st.text(rel)

    # Visualize the full graph as an interactive graph
    with graph_tab:
        nodes, edges = build_graph(schema, mode, use_context)

        config = Config(width=700,
                        height=700,
                        directed=True,
                        physics=False,
                        hierarchical=False)

        agraph(nodes=nodes,
               edges=edges,
               config=config)

    # Visualize the graph statistics for each schema
    with graph_stats_tab:
        # Collect metrics
        all_metrics = {}
        ent_recalls, rel_recalls, degrees = {}, {}, {}

        for schema_any in schemas:
            sents, corefs, rels,  ents,  = load_data(
                schema_any, mode, use_context, grouped=False)
            metrics = get_metrics(ents, rels)

            all_metrics[schema_any] = metrics
            ent_recalls[schema_any] = get_abs_recall_dist(ents)
            rel_recalls[schema_any] = get_abs_recall_dist(rels)
            degrees[schema_any] = get_degrees_dist(ents, rels)

        for v, name in [(ent_recalls, 'entities per sentence'), (rel_recalls, 'relations per sentence'), (degrees, 'degrees of graph')]:
            long_df = to_long_format_df(v, name)

            fig = px.bar(long_df, x="schema", y="value", color=name, color_discrete_sequence=plotly.colors.qualitative.Plotly,
                         title=f"{name} distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame(all_metrics))

    # Explore the graoh in encyclopedic fashion
    with ecyclo_tab:

        # The entry point is a list of all entities
        sents, corefs, rels,  ents,  = load_data(
            schema, mode, use_context, grouped=False)

        rels = [rel for sent in rels for rel in sent]

        rels_dict = defaultdict(list)
        for rel in rels:
            rels_dict[rel[2]].append(rel)

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
                              on_click=set_cur, args=(ent,))
            # Visualization  for specifc item
            else:
                st.button(label='Back', on_click=set_cur,
                          args=(None,), type='primary')
                st.subheader(st.session_state['current_ent'])

                # Get all relations that involve the current entity
                rels_ = [
                    rel for rel in rels if st.session_state['current_ent'] in rel]

                for rel in rels_:
                    col1, col2 = st.columns(2)
                    col1.button(label=rel[2], key=str(
                        uuid.uuid4()), on_click=set_cur, args=(None, rel[2]))
                    col2.button(label=rel[1], key=str(
                        uuid.uuid4()), on_click=set_cur, args=(rel[1],))

        # Relations tab
        if cur_selection == 'relations':
            if st.session_state['current_rel'] == None:
                for rel_name, rel in rels_dict.items():
                    st.button(label=rel_name + ' (' + str(len(rel)) + ' entities) ',
                              on_click=set_cur, args=(None, rel_name,), key=str(uuid.uuid4()))
            else:
                st.button(label='Back', on_click=set_cur,
                          args=(None, None), type='primary')
                st.subheader(st.session_state['current_rel'])

                for rel in rels_dict[st.session_state['current_rel']]:
                    col1, col2, col3 = st.columns(3)
                    col1.button(label=rel[0], key=str(
                        uuid.uuid4()), on_click=set_cur, args=(rel[0],))
                    col2.button(label=rel[2], key=str(
                        uuid.uuid4()), on_click=set_cur, args=(None, rel[2]))
                    col3.button(label=rel[1], key=str(
                        uuid.uuid4()), on_click=set_cur, args=(rel[1],))

        # When clicking on an entity we get a list of all relations that entity is involved in
