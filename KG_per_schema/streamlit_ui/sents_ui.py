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


def viz_sents_ui(schema, mode, use_context):
    '''Visualizes the sentences and their entities and relations.'''
    groups = load_data(schema, mode, use_context, grouped=True)

    if isinstance(groups, dict):
        for idx, group in groups.items():
            st.subheader(idx)

            for sent, ent, rels in group:
                annotated_text(ent)
                for rel in rels:
                    st.text(rel)
