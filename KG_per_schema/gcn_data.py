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

from collections import OrderedDict

import pandas as pd

from transformers import BertTokenizer
from transformers import BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_embeddings(phrases, batch_size=32):
    embeddings = []

    # Creating a progress bar
    progress_bar = tqdm(range(0, len(phrases), batch_size),
                        desc='Processing phrases')

    # Loop over phrases in batches
    for i in progress_bar:
        batch = phrases[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt",
                           padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        batch_embeddings = torch.mean(outputs[0], dim=1)
        embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def process(entitity_sents, rels, ):
    # Append the entity types to the relations of the graph
    for idx, ent_sent in enumerate(entitity_sents):
        for part in ent_sent:
            if isinstance(part, tuple) or isinstance(part, list):
                rels[idx].append((part[0], part[1], 'is a'))

    entity_to_idx = []
    for rel_sent in rels:
        idxs = {}
        entity_to_idx.append(idxs)
        for rel in rel_sent:
            if rel[0] not in idxs:
                idxs[rel[0]] = len(idxs)
            if rel[1] not in idxs:
                idxs[rel[1]] = len(idxs)

    edge_idx = []

    for idx, rel_sent in enumerate(rels):
        # print('\n\n')
        edge_idx_for_sent = np.zeros((len(rel_sent), 2), dtype=np.int64)

        for idx2,  rel in enumerate(rel_sent):
            edge_idx_for_sent[idx2][0] = entity_to_idx[idx][rel[0]]
            edge_idx_for_sent[idx2][1] = entity_to_idx[idx][rel[1]]

        edge_idx_for_sent = np.array(edge_idx_for_sent)
        edge_idx_for_sent = np.transpose(edge_idx_for_sent)
        edge_idx.append(edge_idx_for_sent)

    # Flatten the phrases and get embeddings all at once
    phrases = [key for sent in entity_to_idx for key, val in sent.items()]
    embeddings = get_embeddings(phrases)

    # Flatten the list of tensors into a 2D array
    embeddings_2d = embeddings.detach().numpy()

    # Apply PCA
    pca = PCA(n_components=32)  # reduce to 50 dimensions
    reduced_embeddings = pca.fit_transform(embeddings_2d)
    print('Explained variation per principal component: {}'.format(
        pca.explained_variance_ratio_))

    # Now split the embeddings back into the per-sentence structure
    node_features = []
    start = 0
    for sent in entity_to_idx:
        end = start + len(sent)
        node_features.append(reduced_embeddings[start:end])
        start = end

    return edge_idx, node_features


sents, corefs, rels,  entitity_sents,  = load_data(
    ['ace05'], 'OR', False, grouped=False)

sents_cont, corefs, rels_cont,  entitity_sents_cont,  = load_data(
    ['ace05'], 'OR', True, grouped=False)

num_misses = 0
hits = 0
indices_to_remove = []
for sent in sents:
    try:
        hits += 1
        indices_to_remove.append(sents_cont.index(sent))
    except Exception as e:
        # print(e)
        num_misses += 1

print('misses',  num_misses, 'hits', hits)


# Remove sentences in the true data that are not in the context data
sents_cont = [i for idx, i in enumerate(
    sents_cont) if idx not in indices_to_remove]
rels_cont = [i for idx, i in enumerate(
    rels_cont) if idx not in indices_to_remove]
entitity_sents_cont = [i for idx, i in enumerate(
    entitity_sents_cont) if idx not in indices_to_remove]

# Get connections and node features
edge_idx, node_features = process(entitity_sents, rels)
edge_idx_cont, node_features_cont = process(entitity_sents_cont, rels_cont)


dataset = []
for s, l in zip(edge_idx, node_features):
    if len(l) > 0:
        data = Data(x=torch.Tensor(l), edge_index=torch.Tensor(s),
                    y=torch.Tensor(np.array([1], dtype=np.int64)))
        dataset.append(data)

print(len(dataset))

for s, l in zip(edge_idx_cont, node_features_cont):
    if len(l) > 0:
        data = Data(x=torch.Tensor(l), edge_index=torch.Tensor(s),
                    y=torch.Tensor(np.array([0], dtype=np.int64)))

        # print(data.x)
        dataset.append(data)

print(len(dataset))

with open('KG_per_schema/gcn_data/ace05.pkl', 'wb') as f:
    pickle.dump(dataset, f)
