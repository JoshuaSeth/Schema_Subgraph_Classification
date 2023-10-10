import pickle
from torch_geometric.data import Dataset, download_url
from gcn_data import MyOwnDataset
import glob
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from subgraph_viz_example import visualize_subgraph
from collections import Counter

hash_to_idx = {}


def to_idx(hash):
    global hash_to_idx
    if not hash in hash_to_idx:
        hash_to_idx[hash] = len(hash_to_idx)
    return hash_to_idx[hash]


def hash_data(data):
    x = str(subgraph.x.cpu().numpy().tolist())
    y = str(subgraph.y.cpu().numpy().tolist())
    edge_index = str(subgraph.edge_index.cpu().numpy().tolist())
    edge_types = str(subgraph.edge_types.cpu().numpy().tolist())
    return x + '\n' + y + '\n' + edge_index + '\n' + edge_types


for dataset_fpath in tqdm(glob.glob(f"/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/KG_per_schema/gcn_subgraph_data_filtered/*.pkl")):
    with open(dataset_fpath, 'rb') as f:
        dataset = pickle.load(f)

    data_counter = []

    for subgraph in dataset:
        if to_idx(hash_data(subgraph)) in data_counter:
            visualize_subgraph(sub)
        data_counter.append(to_idx(hash_data(subgraph)))

    data_counter = Counter(data_counter)

    print(dataset_fpath)
    print(data_counter.most_common(10))
