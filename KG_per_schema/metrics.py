'''Several metric functions that can be applied to the loaded data from the results_loader.py file.'''

import subprocess
import glob
import os
from KG_per_schema.utils import project_path
import json
from tqdm import tqdm
from copy import deepcopy
from typing import List
import pickle
from collections import defaultdict
from streamlit_agraph import Node, Edge, Config, agraph
import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
from community import community_louvain

# Some variables for the operation
dygie_prediction_dir_path = project_path + '/KG_per_schema/data/predictions/'
group_info_fpath = project_path + '/KG_per_schema/data/group_info/group_info.pkl'


def to_nx_graph(ents, rels):
    '''Converts a list of entity tagged sentences and relations to a nx graph'''
    G = nx.Graph()

    for rel_sent in rels:
        for rel in rel_sent:
            G.add_edge(rel[0], rel[1], label=rel[2])

    for ent_sent in ents:
        for ent in ent_sent:
            if isinstance(ent, tuple) or isinstance(ent, list):
                G.add_edge(ent[1], ent[0], label='type')
    return G


# @st.cache_data(persist="disk")
def get_metrics(ents, rels):
    '''Gets the metrics for the given ents and rels as a dict.'''
    metrics = {}
    G = to_nx_graph(ents, rels)

    metrics['asbolute recall ner'] = len(ents)
    metrics['asbolute recall re'] = len(rels)

    degrees = [item[1] for item in list(G.degree)]
    metrics['mean degree'] = np.nanmean(degrees)
    metrics['std degree'] = np.nanstd(degrees)

    nx_metrics = {'degree centrality': nx.degree_centrality(G),
                  'closeness centrality': nx.closeness_centrality(G),
                  #   'betweenness centrality': nx.betweenness_centrality(G),
                  #   'pagerank': nx.pagerank(G)
                  'clustering coefficient': nx.clustering(G),

                  }
    for k, v in nx_metrics.items():
        if isinstance(v, dict):
            t = sorted(v.values())

            metrics['mean ' + k] = np.nanmean(t)
            metrics['std ' + k] = np.nanstd(t)
        else:
            metrics[k] = v

    try:
        metrics['modularity'] = community_louvain.modularity(
            community_louvain.best_partition(G), G)
    except:
        pass
    try:
        metrics['avg path length'] = 1  # nx.average_shortest_path_length(G)
    except Exception as e:
        pass
    try:
        metrics['diameter'] = nx.diameter(G)
    except:
        pass
    try:
        metrics['density'] = nx.density(G)
    except:
        pass

    return metrics


def get_degrees_dist(ents, rels):
    '''Gets the degrees distribution of a nx graph constructed from rels and ents For example: with degree 15 there is 1 node, with degree 10 there are 2 nodes, with degree 3 there are 20 nodes, etc.'''
    G = to_nx_graph(ents, rels)
    degrees = defaultdict(int)
    for item in list(G.degree):
        degrees[item[1]] += 1
    return degrees


def get_clusterings(ents, rels):
    G = to_nx_graph(ents, rels)
    result = defaultdict(int)
    for item in list(nx.clustering(G).values()):
        result[item] += 1
    return result


def to_long_format_df(items: dict[dict], keyname='metric') -> pd.DataFrame:
    '''Given a dict of dicts or defaultdicts of the for example the degree distribution or number of sentences with a certain triple it will return a long format dataframe wherea  row is [schema, keyname, value]. Can be used with plotting libraries for stacked bar plots.'''
    rows = []
    for schema_name, values in items.items():
        for k, v in values.items():
            rows.append([schema_name, k, v])

    df = pd.DataFrame(rows, columns=['schema', keyname, 'value'])

    df.sort_values(by=[keyname], inplace=True)

    df[keyname] = df[keyname].astype(str)

    return df


def get_abs_recall_dist(items):
    '''Get the distribution of absolute recalls. For example: 3 sentences with 4 triples, 5 sentences with 3 triples, 20 sentences with 2 triples and 100 sentences with 1 triple. Can be given an sentence with entities or relations.'''
    distribution = defaultdict(int)

    # Entity sentence
    if len(items) > 0:
        print('getting abs recall')
        is_entities = False
        for sent in items:
            if len(sent) > 0:
                if (isinstance(sent[0], tuple) and len(sent[0]) == 2) or isinstance(sent[0], str):
                    is_entities = True
                    break
        if is_entities:
            print('are ents')
            for sent in items:
                num_ents = len(
                    [part for part in sent if isinstance(part, tuple)])
                distribution[num_ents] += 1
        # Relation sentence
        else:
            print('are rels')
            for sent in items:
                num_rels = len(sent)
                distribution[num_rels] += 1

    return distribution
