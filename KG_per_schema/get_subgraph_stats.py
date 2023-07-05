import os
import glob
from tqdm import tqdm
from gcn_data import MyOwnDataset
import pickle
from torch_geometric.utils import to_networkx
import networkx as nx
from community import community_louvain
import numpy as np
import pandas as pd

datasets_path = '/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/KG_per_schema/gcn_subgraph_data/'

results = []

metric_names = ['degrees',
                'degrees std',
                'degree_centralities',
                'degree_centralities std',
                'closeness_centralities',
                'closeness_centralities std',
                'clusterings',
                'clusterings std',
                'modularities',
                'modularities std',
                'avg_path_lens',
                'avg_path_lens std',
                'densities',
                'densities std',
                'schema',
                'mode',
                ]

for dataset_fpath in tqdm(glob.glob(f"{datasets_path}*.pkl")):
    degrees = []
    degree_centralities = []
    closeness_centralities = []
    clusterings = []
    modularities = []
    avg_path_lens = []
    densities = []

    properties = os.path.basename(dataset_fpath).split('_')
    schema_name = properties[0]
    mode = properties[1]

    with open(dataset_fpath, 'rb') as f:
        dataset = pickle.load(f)

    for data in dataset.data_list:

        # Assuming you have a PyTorch Geometric data object named `data`
        G = to_networkx(data, to_undirected=True)

        degrees.append(np.nanmean(list(G.degree())))
        degree_centralities.append(np.nanmean(
            list(nx.degree_centrality(G).values())))
        closeness_centralities.append(np.nanmean(
            list(nx.closeness_centrality(G).values())))

        clustering = nx.clustering(G)
        if isinstance(clustering, dict):
            clusterings.append(np.nanmean(list(clustering.values())))
        else:
            clusterings.append(clustering)

        modularities.append(community_louvain.modularity(
            community_louvain.best_partition(G), G))

        try:
            avg_path_lens.append(nx.average_shortest_path_length(G))
        except:
            pass

        densities.append(nx.density(G))

    stats = [np.nanmean(degrees),
             np.nanstd(degrees),
             np.nanmean(degree_centralities),
             np.nanstd(degree_centralities),
             np.nanmean(closeness_centralities),
             np.nanstd(closeness_centralities),
             np.nanmean(clusterings),
             np.nanstd(clusterings),
             np.nanmean(modularities),
             np.nanstd(modularities),
             np.nanmean(avg_path_lens),
             np.nanstd(avg_path_lens),
             np.nanmean(densities),
             np.nanstd(densities)]

    for idx, stat in enumerate(stats):
        results.append([metric_names[idx], stat, schema_name, mode])


# Convert the results dictionary to a DataFrame
df = pd.DataFrame(results, columns=['metric', 'value', 'schema', 'mode'])

df_pivot = df.pivot_table(index=['metric'],
                          columns=['schema', 'mode'],
                          values='value')

df_pivot = df_pivot.style.format(decimal='.', thousands=',', precision=2)

print(df_pivot)

# Transpose the DataFrame so that the statistic names are the index and the schemas are the column names
# df = df.transpose()

print(df_pivot.to_latex())
