'''SUbgraph vidualizer to check whether they make snese. With the original data loader.'''

import pickle
import json
from results_loader import load_data, build_graph
import networkx as nx
import matplotlib.pyplot as plt
from metrics import to_nx_graph
from random import shuffle


def visualize_subgraph(rels, ents):
    shuffle(rels)
    shuffle(ents)

    # Create networkx graph from data
    G = to_nx_graph(ents, rels)

    for edge in G.edges(data=True):
        print(edge)

    # Define position - let's use a circular layout for this example
    pos = nx.kamada_kawai_layout(G)

    # Draw the graph with nodes in orange, edges in gray
    nx.draw(G, pos, node_color='orange', edge_color='gray',
            node_size=500, with_labels=False)

    # Draw node labels with a specified font
    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12,
                            font_weight='bold', font_family='sans-serif')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color='blue')

    plt.margins(0.3)

    plt.show()


def visualize_some_subgraph(schema, mode, use_context, is_research, subgraph_idx):
    sents, corefs, rels,  ents,  = load_data(
        [schema], mode, use_context, is_research=is_research, index=0, grouped=False)

    sents = [sents[subgraph_idx]]
    # corefs = [corefs[subgraph_idx]]
    rels = [rels[subgraph_idx]]
    ents = [ents[subgraph_idx]]

    visualize_subgraph(rels, ents)


if __name__ == '__main__':
    visualize_some_subgraph('scierc', 'OR', False, True, 0)
