import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import glob
import subprocess
from KG_per_schema.utils import project_path, get_model_fname
from KG_per_schema.results_loader import load_data, build_graph
from KG_per_schema.metrics import get_metrics, get_clusterings,  get_abs_recall_dist, get_degrees_dist, to_long_format_df
from annotated_text import annotated_text
import pickle
from collections import defaultdict
from streamlit_agraph import agraph, Config, Node, Edge
import kaleido  # required

import plotly.express as px
import plotly
import uuid


def viz_graph_stats_ui(schemas, mode, use_context):
    '''Visualizes the graph statistics such as degree and absolute recall in several plots.'''
    # Collect metrics
    all_metrics = {}
    ent_recalls, rel_recalls, degrees, clusterings = {}, {}, {}, {}

    for schema_any in schemas:
        sents, corefs, rels,  ents,  = load_data(
            [schema_any], mode, use_context, is_research=False, grouped=False)
        sents2, corefs2, rels2,  ents2,  = load_data(
            [schema_any], mode, use_context, is_research=False, grouped=False)
        sents.extend(sents2)
        corefs.extend(corefs2)
        rels.extend(rels2)
        ents.extend(ents2)

        metrics = get_metrics(ents, rels)

        all_metrics[schema_any] = metrics
        ent_recalls[schema_any] = get_abs_recall_dist(ents)
        rel_recalls[schema_any] = get_abs_recall_dist(rels)
        degrees[schema_any] = get_degrees_dist(ents, rels)
        clusterings[schema_any] = get_clusterings(ents, rels)

        # Visualize a distribution of absolute recall

        for naming, metric in zip(['Number of entities detected', 'Number of relations detected', 'Number of degrees per node', 'Clustering coefficient per node'], [ent_recalls, rel_recalls, degrees, clusterings]):
            df = pd.DataFrame(columns=[naming, 'Number of items'])
            df[naming] = metric[schema_any].keys()
            df['Number of items'] = metric[schema_any].values()
            # df.set_index(naming, inplace=True)
            # df = df[df.index < 50]
            if len(df) > 0:
                try:
                    # If the values are floats
                    if df[naming].dtype == np.float64:
                        nbins = 20
                    else:
                        nbins = int(df[naming].max())+1
                except:
                    nbins = 10
                fig = px.histogram(
                    df, x=naming, y='Number of items', title=f"{naming} distribution for {schema_any}", nbins=nbins)
                fig.update_layout(
                    yaxis_title="Number of items",
                    xaxis_title=f"{naming}",
                    font=dict(
                        family="Arial",
                        size=18,
                        color="RebeccaPurple"
                    )
                )
                if df['Number of items'].mean() / df['Number of items'].max() < 0.1:
                    fig.update_yaxes(
                        title_text="Number of items (logarithmic)", type="log")

                fig.write_image("/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/dist_figs/" +
                                f"{naming} distribution for {schema_any}.png", width=1920, height=1080, format="png", engine='orca')

                st.plotly_chart(fig, use_container_width=True)

    for v, name in [(ent_recalls, 'entities per sentence'), (rel_recalls, 'relations per sentence'), (degrees, 'degrees of graph')]:
        long_df = to_long_format_df(v, name)

        fig = px.bar(long_df, x="schema", y="value", color=name, color_discrete_sequence=plotly.colors.qualitative.Plotly,
                     title=f"{name} distribution")
        st.plotly_chart(fig, use_container_width=True)

    grand_df = pd.DataFrame(all_metrics)
    grand_df = grand_df.style.format(decimal='.', thousands=',', precision=2)
    # grand_df = grand_df.round(2)
    print(grand_df.to_latex())
    st.dataframe(grand_df)
