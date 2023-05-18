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
current_ent = None


def show_connections(ent):
    global current_ent
    current_ent = ent


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

        ents = set([
            part[0] for sent in ents for part in sent if isinstance(part, tuple)])

        if not current_ent:
            for ent in ents:
                st.button(label=ent, on_click=show_connections, args=(ent,))
        else:
            st.text(current_ent)
        # When clicking on an entity we get a list of all relations that entity is involved in
