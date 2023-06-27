
from results_loader import load_data, build_graph
import matplotlib.pyplot as plt
from netgraph import Graph
from metrics import to_nx_graph
import networkx as nx
from community import community_louvain
import numpy as np
import plotly.graph_objects as go


schema = 'genia'
mode = 'OR'
use_context = True

sents, corefs, rels,  ents,  = load_data(
    [schema], mode, use_context, grouped=False)
G = to_nx_graph(ents, rels)

best_partition = community_louvain.best_partition(G)

community_ids = list(set(best_partition.values()))
community_colors = plt.cm.rainbow(np.linspace(0, 1, len(community_ids)))
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


# draw the graph
nx.draw(G, pos, node_color=node_color, alpha=0.8, edge_color='gray',
        node_size=17, with_labels=False)


# # infer the best partition using Louvain
# partition = community_louvain.best_partition(G)

# # get unique groups
# groups = set(partition.values())
# group_dict = {v: i for i, v in enumerate(groups)}

# # create a list of central positions for nodes in each partition to be used in the hovertemplate
# central_pos = nx.spring_layout(G)

# for node in G.nodes:
#     G.nodes[node]['group'] = group_dict[partition[node]]
#     G.nodes[node]['pos'] = central_pos[node]

# # create edges
# edge_x = []
# edge_y = []
# for edge in G.edges():
#     x0, y0 = G.nodes[edge[0]]['pos']
#     x1, y1 = G.nodes[edge[1]]['pos']
#     edge_x.extend([x0, x1, None])
#     edge_y.extend([y0, y1, None])

# # create nodes
# node_x = [pos[0] for pos in central_pos.values()]
# node_y = [pos[1] for pos in central_pos.values()]
# node_group = [data['group'] for data in G.nodes.values()]

# # plot the graph using plotly
# edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(
#     width=0.5, color='lightgrey'), hoverinfo='none', mode='lines')

# node_trace = go.Scatter(x=node_x, y=node_y,
#                         mode='markers',
#                         hoverinfo='text',
#                         marker=dict(showscale=True, colorscale='Rainbow', reversescale=True, color=[], size=10, colorbar=dict(thickness=15, title='Node Connection', xanchor='left', titleside='right'),
#                                     line_width=2))

# node_adjacencies = []
# for node, adjacencies in enumerate(G.adjacency()):
#     node_adjacencies.append(len(adjacencies[1]))

# node_trace.marker.color = node_group
# node_trace.text = [
#     f'Node: {node}<br># of connections: {adj}' for node, adj in enumerate(node_adjacencies)]

# fig = go.Figure(data=[edge_trace, node_trace],
#                 layout=go.Layout(
#                     title="Network graph made with Python",
#                     titlefont_size=16,
#                     showlegend=False,
#                     hovermode='closest',
#                     margin=dict(b=20, l=5, r=5, t=40),
#                     xaxis=dict(showgrid=False, zeroline=False,
#                                showticklabels=False),
#                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                 )
# fig.show()


plt.show()
