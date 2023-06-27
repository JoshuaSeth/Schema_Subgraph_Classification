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
import matplotlib.pyplot as plt
from netgraph import Graph, InteractiveGraph, EditableGraph
from metrics import to_nx_graph
# Cannot be memoized
# @st.cache_data(persist="disk", experimental_allow_widgets=True)


def viz_graph_ui(schema, mode, use_context):
    '''Visualizes the complete interactive graph.'''
    nodes, edges = build_graph(schema, mode, use_context)

    config = Config(width=700,
                    height=700,
                    directed=True,
                    physics=False,
                    hierarchical=False)

    agraph(nodes=nodes,
           edges=edges,
           config=config)
