
from results_loader import load_data, build_graph
import matplotlib.pyplot as plt
from netgraph import Graph
from metrics import to_nx_graph
import networkx as nx
from community import community_louvain
import numpy as np
import plotly.graph_objects as go


for schema in ['scierc', 'None', 'genia', 'covid-event', 'ace05', 'ace-event']:
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

            best_partition = community_louvain.best_partition(G)

            community_ids = list(set(best_partition.values()))
            community_colors = plt.cm.rainbow(
                np.linspace(0, 1, len(community_ids)))
            community_id_to_color = {
                community_id: community_colors[i] for i, community_id in enumerate(community_ids)}

            # Assign each node a color based on its community, and add some random noise
            node_color = []
            for node in G.nodes:
                base_color = community_id_to_color[best_partition[node]]
                noise = (np.random.rand(3) - 0.5) * 0.35
                node_color.append(np.clip(base_color[:3] + noise, 0, 1))

            # generate a layout for the nodes
            pos = nx.spring_layout(G)
            # Optional: pos = nx.spectral_layout(G)

            # draw the graph
            fig = nx.draw(G, pos, node_color=node_color, alpha=0.8, edge_color='gray',
                          node_size=17, with_labels=False)

            plt.show()
            plt.savefig(
                '/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/KG_per_schema/network_graphs/' + f"{schema} {mode}.png", bbox_inches='tight')
