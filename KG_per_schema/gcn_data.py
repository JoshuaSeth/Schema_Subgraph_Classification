'''Convert data to a toch scatter compatible dataset'''
from tqdm import tqdm
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
# import plotly.express as px
# import plotly
import uuid
from streamlit_ui.encyclo_ui import build_encyclo_data
import torch
from torch_geometric.data import Data
import random
# from gensim.models import KeyedVectors
import os
from sklearn.decomposition import PCA
from metrics import to_nx_graph

from collections import OrderedDict

import pandas as pd

from transformers import BertTokenizer
from transformers import BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

schema_name = 'scierc'

with open('/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/KG_per_schema/scierc_init_embeddings.pkl', 'rb') as f:
    initial_node_embeddings = pickle.load(f)


def build_graph_part(research_sents_or_not):
    sents, corefs, rels,  entitity_sents,  = load_data(
        [schema_name], 'OR', False, is_research=research_sents_or_not,  grouped=False)
    print(len(sents))
    G = to_nx_graph(entitity_sents, rels)

    global_entity_to_index = {
        entity: index for index, entity in enumerate(G.nodes())}

    # Append the entity types to the relations of the graph
    for idx, ent_sent in enumerate(entitity_sents):
        for part in ent_sent:
            if isinstance(part, tuple) or isinstance(part, list):
                rels[idx].append((part[0], part[1], 'is a'))

    dataset = []
    for rel_sent in rels:
        if len(rel_sent) > 0:
            edge_indices = []

            local_entity_to_index = {}
            for rel in rel_sent:
                if rel[0] not in local_entity_to_index:
                    local_entity_to_index[rel[0]] = len(local_entity_to_index)
                if rel[1] not in local_entity_to_index:
                    local_entity_to_index[rel[1]] = len(local_entity_to_index)

            for rel in rel_sent:
                edge_indices.append(
                    [local_entity_to_index[rel[0]], local_entity_to_index[rel[1]]])
            edge_indices = np.array(edge_indices).transpose()

            node_features = []
            for rel in rel_sent:
                node_features.append(
                    initial_node_embeddings[global_entity_to_index[rel[0]]])
                node_features.append(
                    initial_node_embeddings[global_entity_to_index[rel[1]]])
            node_features = np.array(node_features)

            data = Data(x=torch.Tensor(node_features), edge_index=torch.Tensor(edge_indices),
                        y=torch.Tensor(np.array([int(research_sents_or_not)], dtype=np.int64)))
            dataset.append(data)

    return dataset


data = build_graph_part(research_sents_or_not=True)
data.extend(build_graph_part(research_sents_or_not=False))

# del edge_idx
# del node_features


# print(len(dataset))

# edge_idx_cont, node_features_cont = process(
#     entitity_sents_cont, rels_cont)

# del entitity_sents_cont
# del rels_cont

# for s, l in zip(edge_idx_cont, node_features_cont)[:-70]:
#     if len(l) > 0:
#         data = Data(x=torch.Tensor(l), edge_index=torch.Tensor(s),
#                     y=torch.Tensor(np.array([0], dtype=np.int64)))

#         # print(data.x)
#         dataset.append(data)

# print(len(dataset))

with open(f'KG_per_schema/gcn_data/{schema_name}.pkl', 'wb') as f:
    pickle.dump(data, f)
