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
from streamlit_agraph import agraph, Config, Node, Edge


@st.cache_data(persist="disk", experimental_allow_widgets=True)
def build_graph(schema: str, mode: str = 'AND', context: bool = False, index=None):
    '''Wrapper around the load_data method. Builds a graph from the data which can be used in streamlit. The nodes and edges are used to build the graph with: agraph(nodes=nodes, edges=edges, config=config) Grouping makes no difference for building the graph.

    Parameters
    ------------
        schema: str
            Which schema to use. One of: [scierc, None (= mechanic coarse), genia, covid-event (= mechanic granular), ace05, ace-event]
        mode: str, Optional
            Whether to use sentences that are a research challenge or direction or are both. One of: [AND, OR]. Default: AND
        context: bool, Optional
            Whether to include context sentences or not. Default: False
        index: int, Optional
            Whether to use a specific index file. If None then all data conforming to the request params is used. If given an index only a single datafile containing the request params and this specific index is used. Which might be handy for taking small samples. Default: None
        pbar: st pbar instance, Optional
            A streamlit progress bar instance. If given the progress bar will be updated. Default: None

    Return
    -----------
        nodes, edges: list, list
            Returns a list of nodes, a list of edges and a config object.'''


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
        pbar = st.progress(0, text='Building the graph...')
        # container = st.container()

        sents, corefs, rels, ents = load_data(
            schema, mode, context=False, index=None, grouped=False)

        nodes = []
        edges = []
        ids = set()
        for idx, ent_sent in enumerate(ents):
            for item in ent_sent:
                if isinstance(item, tuple):
                    if not item[0] in ids:
                        nodes.append(
                            Node(id=item[0], label=item[0] + ' (' + item[1] + ')', size=25,))
                    ids.add(item[0])

            if pbar:
                prog = idx*0.9 / (len(ents)*0.9 + len(rels)*0.1)
                pbar.progress(prog, text='Adding entities...')

        for idx, rel_sent in enumerate(rels):
            for rel in rel_sent:
                edges.append(Edge(source=rel[0], label=rel[2], target=rel[1]))

            if pbar:
                prog = (len(ents)*0.9+idx*0.1) / \
                    (len(ents)*0.9 + len(rels)*0.1)

                pbar.progress(prog, text='Adding relations...')

        config = Config(width=700,
                        height=700,
                        directed=True,
                        physics=False,
                        hierarchical=False)

        agraph(nodes=nodes,
               edges=edges,
               config=config)

        pbar = None
