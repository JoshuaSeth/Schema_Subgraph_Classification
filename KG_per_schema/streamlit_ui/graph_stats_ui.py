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


def viz_graph_stats_ui(schemas, mode, use_context):
    '''Visualizes the graph statistics such as degree and absolute recall in several plots.'''
    # Collect metrics
    all_metrics = {}
    ent_recalls, rel_recalls, degrees = {}, {}, {}

    for schema_any in schemas:
        sents, corefs, rels,  ents,  = load_data(
            [schema_any], mode, use_context, grouped=False)
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
