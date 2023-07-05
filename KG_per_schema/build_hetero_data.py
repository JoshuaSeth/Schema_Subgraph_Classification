'''Builds a torch geometric graph object from the relations and sentences.'''

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch
from results_loader import load_data, build_graph
import matplotlib.pyplot as plt
from netgraph import Graph
from metrics import to_nx_graph
import networkx as nx
from community import community_louvain
import numpy as np
import plotly.graph_objects as go
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import HeteroData
from collections import defaultdict
import torch_geometric.transforms as T

schema = 'scierc'
mode = 'OR'
use_context = False

for schema in tqdm(['scierc', 'None', 'genia', 'covid-event', 'ace05', 'ace-event']):
    for mode in ['OR', 'AND']:
        # Load data with internal data loder module
        sents, corefs, rels,  ents,  = load_data(
            [schema], mode, use_context, is_research=True, grouped=False)
        sents_false, corefs_false, rels_false,  ents_false,  = load_data(
            [schema], mode, use_context, is_research=False, grouped=False)

        ents.extend(ents_false)
        rels.extend(rels_false)

        # Create networkx graph from data
        G = to_nx_graph(ents, rels)

        # Create a mapping of nodes to indices
        node_to_index = {node: index for index, node in enumerate(G.nodes())}

        # Get edge indices connection per edge type
        edges_as_indices = defaultdict(list)
        for edge in G.edges(data=True):
            edge_type = edge[2]['label']
            # Add the indices of the two connected nodes
            edges_as_indices[edge_type].append(
                [node_to_index[edge[0]], node_to_index[edge[1]]])

        # Get the unique edge labels
        edge_types = list(set([edge[2]['label']
                          for edge in G.edges(data=True)]))

        # Initialize node features randomly
        num_nodes = len(G.nodes())
        num_node_features = 5
        x = torch.randn((num_nodes, num_node_features))

        data = HeteroData()
        data['entity'].x = x
        for edge_type in edge_types:
            # Get the edge indices for this edge type
            data['entity', edge_type, 'entity'].edge_index = torch.tensor(
                np.array(edges_as_indices[edge_type]).transpose(), dtype=torch.long)

        print(data)

        torch.save(
            data, '/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/full_schema_node_embeddings/' + f'{schema}_{mode}.pth')
