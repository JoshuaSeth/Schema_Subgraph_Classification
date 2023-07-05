
from results_loader import load_data, build_graph
import matplotlib.pyplot as plt
from netgraph import Graph
from metrics import to_nx_graph
import networkx as nx
from community import community_louvain
import numpy as np
import plotly.graph_objects as go

for schema in ['None', 'scierc', 'genia', 'covid-event', 'ace05', 'ace-event']:
    for mode in ['OR', 'AND']:
        for use_context in [False]:
            sents, corefs, rels,  ents,  = load_data(
                [schema], mode, False, is_research=True, grouped=False)
            sents2, corefs2, rels2,  ents2,  = load_data(
                [schema], mode, False, is_research=False, grouped=False)
            sents.extend(sents2)
            corefs.extend(corefs2)
            rels.extend(rels2)
            ents.extend(ents2)
            G = to_nx_graph(ents, rels)

            print(schema, len(G.nodes()), len(G.edges()))
