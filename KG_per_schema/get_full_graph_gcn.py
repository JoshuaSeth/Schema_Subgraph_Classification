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

        # Create a LabelEncoder for the edge labels
        label_encoder = LabelEncoder()

        # Now create your list of edges, with nodes represented by their indices
        edges_as_indices = []
        edge_labels = []
        for edge in G.edges(data=True):
            edges_as_indices.append(
                [node_to_index[edge[0]], node_to_index[edge[1]]])
            edge_labels.append(edge[2]['label'])  # 'label' is the key

        edges_as_indices = np.array(edges_as_indices).transpose()

        # Convert numpy array to tensor of type long
        edges_as_indices_tensor = torch.tensor(
            edges_as_indices, dtype=torch.long)

        # Fit the LabelEncoder and transform the edge labels
        edge_labels_encoded = label_encoder.fit_transform(edge_labels)

        # Convert edge_labels_encoded to a tensor
        edge_labels_tensor = torch.tensor(
            edge_labels_encoded, dtype=torch.long)

        # Assuming you have edge_index for the full graph
        num_nodes = edges_as_indices_tensor.shape[1]
        num_node_features = 5

        # Initialize node features randomly
        x = torch.randn((num_nodes, num_node_features))

        # # Use GCN layer for message passing
        # conv1 = GCNConv(num_node_features, 5)  # Reduce to 2 features
        # # pass tensor instead of np array
        # x = conv1(x, edges_as_indices_tensor)
        # # pass tensor instead of np array
        # x = conv1(x, edges_as_indices_tensor)

        # Now x holds your new node features after applying one layer of GCN.
        # These are your initial node embeddings for the full graph
        # initial_node_embeddings = x.detach().numpy()

        # print(initial_node_embeddings.shape)

        data = Data(x=x, edge_index=edges_as_indices_tensor,
                    edge_attr=edge_labels_tensor)

        print(f"Number of attributes on data: {len(data)}")
        print(f"Number of nodes: {data.x.shape[0]}")
        print(f"Number of node features: {data.x.shape[1]}")
        print(f"Number of edges: {data.edge_index.shape[1]}")

        torch.save(
            data, '/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/full_schema_node_embeddings/' + f'{schema}_{mode}.pth')
