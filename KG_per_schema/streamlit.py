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
from metrics import get_metrics, get_abs_recall_dist, get_degrees_dist
from annotated_text import annotated_text
import pickle
from collections import defaultdict
from streamlit_agraph import agraph, Config, Node, Edge
import plotly.express as px


# Variables
schemas = ['scierc', 'None', 'genia', 'covid-event', 'ace05', 'ace-event']
modes = ['AND', 'OR']
visualizations = ['sentences', 'graph', 'graph stats']

# Default interface options
st.header('NER & RE Parsing Results')

st.markdown(
    'The analysis of these results can be found in the [research notes](https://docs.google.com/document/d/1i5xHfUvWKcGeX7D1r3Eb1IPm4Bg83-Y0/edit#bookmark=id.jb6w6xm4vqf2).')

schema = st.selectbox('Schema', schemas)
mode = st.selectbox('Mode', modes)
use_context = st.checkbox('Use context', value=True)
sent_tab, graph_tab, graph_stats_tab = st.tabs(visualizations)


if schema != None and mode != None:
    with sent_tab:
        groups = load_data(schema, mode, use_context, grouped=True)

        if isinstance(groups, dict):
            for idx, group in groups.items():
                st.subheader(idx)

                for sent, ent, rels in group:
                    annotated_text(ent)
                    for rel in rels:
                        st.text(rel)

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

    with graph_stats_tab:
        # Collect metrics
        all_metrics = {}
        ent_recalls, rel_recalls, degrees = {}, {}, {}

        for schema in schemas:
            sents, corefs, rels,  ents,  = load_data(
                schema, mode, use_context, grouped=False)
            metrics = get_metrics(ents, rels)

            all_metrics[schema] = metrics
            ent_recalls[schema] = get_abs_recall_dist(ents)
            rel_recalls[schema] = get_abs_recall_dist(rels)
            degrees[schema] = get_degrees_dist(ents, rels)

        st.dataframe(pd.DataFrame(all_metrics))


def to_nx_graph(ents, rels):
    '''Converts a list of entity tagged sentences and relations to a nx graph'''
    G = nx.Graph()

    for rel_sent in rels:
        for rel in rel_sent:
            G.add_edge(rel[0], rel[1], label=rel[2])

    for ent_sent in ents:
        for ent in ent_sent:
            G.add_edge(ent[1], ent[0], label='type')
    return G
