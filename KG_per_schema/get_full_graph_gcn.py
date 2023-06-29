'''Builds the full graph of concatenated research and non-research sentences for all schema's. Can be used for intial node embeddings.'''

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


schema = 'scierc'
mode = 'OR'
use_context = False

for schema in tqdm(['scierc', 'None', 'genia', 'covid-event', 'ace05', 'ace-event']):
    for mode in ['OR', 'AND']:
        sents, corefs, rels,  ents,  = load_data(
            [schema], mode, use_context, is_research=True, grouped=False)
        sents_false, corefs_false, rels_false,  ents_false,  = load_data(
            [schema], mode, use_context, is_research=False, grouped=False)

        ents.extend(ents_false)
        rels.extend(rels_false)

        G = to_nx_graph(ents, rels)

        # Create a mapping of nodes to indices
        node_to_index = {node: index for index, node in enumerate(G.nodes())}

        # Now create your list of edges, with nodes represented by their indices
        edges_as_indices = [[node_to_index[node]
                            for node in edge] for edge in G.edges()]

        edges_as_indices = np.array(edges_as_indices).transpose()

        # Convert numpy array to tensor of type long
        edges_as_indices_tensor = torch.tensor(
            edges_as_indices, dtype=torch.long)

        # Assuming you have edge_index for the full graph
        num_nodes = edges_as_indices_tensor.shape[1]
        num_node_features = 5

        # Initialize node features randomly
        x = torch.randn((num_nodes, num_node_features))

        # Use GCN layer for message passing
        conv1 = GCNConv(num_node_features, 5)  # Reduce to 2 features
        # pass tensor instead of np array
        x = conv1(x, edges_as_indices_tensor)
        # pass tensor instead of np array
        x = conv1(x, edges_as_indices_tensor)

        # Now x holds your new node features after applying one layer of GCN.
        # These are your initial node embeddings for the full graph
        initial_node_embeddings = x.detach().numpy()

        print(initial_node_embeddings.shape)

        with open(f'{schema}_{mode}_init_embeddings.pkl', 'wb') as f:
            pickle.dump(initial_node_embeddings, f)
