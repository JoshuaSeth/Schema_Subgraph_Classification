'''Convert data to a torch scatter compatible dataset. Sues the full_graph for each schema as initial embeddings for this.'''
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
import random
# from gensim.models import KeyedVectors
import os
from sklearn.decomposition import PCA
from metrics import to_nx_graph

from collections import OrderedDict
from random import shuffle
import pandas as pd

from transformers import BertTokenizer
from transformers import BertModel

import os.path as osp

import torch
from torch_geometric.data import Dataset, Data, download_url


from utils import hash_data


class MyOwnDataset(Dataset):
    def __init__(self, data: list[Data], transform=None, pre_transform=None, num_relations=None):
        super(MyOwnDataset, self).__init__('', transform, pre_transform)
        self.data_list = data
        self.num_relations = num_relations

    @property
    def raw_file_names(self):
        return []  # This dataset has no raw files

    @property
    def processed_file_names(self):
        return []  # This dataset has no processed files

    def download(self):
        pass  # This dataset has no download

    def process(self):
        pass  # This dataset doesn't need processing

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def build_graph_part(research_sents_or_not, mode, schema_name):
    # Load the data
    sents, corefs, rels,  entitity_sents,  = load_data(
        [schema_name], mode, False, is_research=research_sents_or_not,  grouped=False)

    print('##########', schema_name, '##########')

    # Create an nx graph from relations and types
    G = to_nx_graph(entitity_sents, rels)

    assert len(entitity_sents) == len(
        rels), f'The number of sentences and relations should be equal. Difference is {len(entitity_sents)} - {len(rels)}'

    global_entity_to_index = {
        entity: index for index, entity in enumerate(G.nodes())}

    # Append the entity types to the relations of the graph
    for idx, ent_sent in enumerate(entitity_sents):
        for part in ent_sent:
            if isinstance(part, tuple) or isinstance(part, list):
                rels[idx].append((part[0], part[1], 'is a'))

    # Create a dict with every possible edge label and an unique index
    edge_label_to_index = {}
    edge_label_to_index['is a'] = 0
    for rel_sent in rels:
        for rel in rel_sent:
            if rel[2] not in edge_label_to_index:
                edge_label_to_index[rel[2]] = len(edge_label_to_index)

    dataset = []
    for rel_sent in rels:
        if len(rel_sent) > 0:
            edge_indices = []
            edge_types_indices = []

            # Create the indices for local entities (entity 1 = 0, entity 2 = 1, entity 3 = 2, etc.)
            local_entity_to_index = {}
            for rel in rel_sent:
                if rel[0] not in local_entity_to_index:
                    local_entity_to_index[rel[0]] = len(local_entity_to_index)
                if rel[1] not in local_entity_to_index:
                    local_entity_to_index[rel[1]] = len(local_entity_to_index)

            # Create indices for the local edges using the local entity indices
            for rel in rel_sent:
                edge_indices.append(
                    [local_entity_to_index[rel[0]], local_entity_to_index[rel[1]]])
                # Use global indices for edge types/labels
                edge_types_indices.append(edge_label_to_index[rel[2]])
            edge_indices = np.array(edge_indices, dtype=np.int64).transpose()
            edge_types_indices = np.array(edge_types_indices, dtype=np.int64)

            # Node features should be read from the global embeddings
            node_features = []
            for rel in rel_sent:
                node_features.append(
                    initial_node_embeddings[global_entity_to_index[rel[0]]])
                node_features.append(
                    initial_node_embeddings[global_entity_to_index[rel[1]]])
            node_features = np.array(node_features)

            data = Data(x=torch.FloatTensor(node_features),
                        edge_index=torch.LongTensor(edge_indices),
                        edge_types=torch.LongTensor(edge_types_indices),
                        y=torch.LongTensor([int(research_sents_or_not)]))

            dataset.append(data)

    return dataset, len(edge_label_to_index)


if __name__ == '__main__':

    embeddings_dir = '/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/KG_per_schema/embeddings_gdrive/'

    for full_graph_init_embeddings_fpath in tqdm(glob.glob(f"{embeddings_dir}*.pth")):
        # Read config properties of this full graph embedding path
        print(full_graph_init_embeddings_fpath)
        properties = os.path.basename(
            full_graph_init_embeddings_fpath).split('_')
        schema_name = properties[0]

        # Tempoarary for the full schema update
        # if schema_name == 'scierc':

        mode = properties[1]
        print('processing', schema_name, mode)

        # Open intial embeddings, set them as variable
        initial_node_embeddings = torch.load(
            full_graph_init_embeddings_fpath, map_location=torch.device('cpu'))
        if torch.isnan(initial_node_embeddings.x).any() or torch.isinf(initial_node_embeddings.x).any():
            print('nan or inf in initial node embeddings')

        global_edge_types = initial_node_embeddings.edge_types
        initial_node_embeddings = initial_node_embeddings.x

        # Build the data from the non-research and research sentences
        data, num_rels = build_graph_part(
            research_sents_or_not=True, mode=mode, schema_name=schema_name)
        secondpart, _ = build_graph_part(
            research_sents_or_not=False, mode=mode, schema_name=schema_name)
        data.extend(secondpart)

        # Filter the subgraphs for duplicates
        filtered_data = []
        unique_datas = set()
        for d in data:
            if hash_data(d) not in unique_datas:
                filtered_data.append(d)
                unique_datas.add(hash_data(d))

        shuffle(filtered_data)
        print('number of subgraphs', len(data),
              'fitered:',  len(filtered_data))

        for data in filtered_data:
            assert data.edge_index.max(
            ) < data.num_nodes, 'Edge node index is larger than number of nodes.'

        # Create the dataset with custom dataset object and the subgraphs as data
        dataset = MyOwnDataset(data=filtered_data, num_relations=num_rels)
        print(dataset)

        with open(f'/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/KG_per_schema/gcn_subgraph_data_filtered/{schema_name}_{mode}.pkl', 'wb') as f:
            pickle.dump(dataset, f)
