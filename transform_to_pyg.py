import pickle
from torch_geometric.utils import from_networkx
from tqdm.auto import tqdm
import torch

def loadGraphs(filepath):
    graphs = []
    for g in pickle.load(open(filepath, "rb")).values():
        data = from_networkx(g)
        
        # node_feats = []
        # for key, val in g.__dict__.items():
        #     if isinstance(val, torch.Tensor) and val.size(0) == g.number_of_nodes():
        #         node_feats.append(val.view(g.number_of_nodes(), -1).float())
        # if node_feats:
        #     data.x = torch.cat(node_feats, dim=1)
        # else:
        data.x = data.nbIp.view(-1, 1).float()  # fallback

        graphs.append(data)
    return graphs


def label_snapshots(num_graphs, anomaly_center_idx, anomaly_window_size=30):
    labels = [0] * num_graphs
    start = max(0, anomaly_center_idx)
    end = num_graphs 
    for i in range(start, end):
        labels[i] = 1
    return labels


def loadData(folder, anomaly_window_size=30):
    d = {}

    names = ["TTNet", "IndoSat", "TM", "AWS", "Google", "ChinaTelecom", "India"]
    nodes = ["9121", "4761", "4788", "200759", "15169", "21217", "55410"]

    for n, node in tqdm(zip(names, nodes), total=len(names)):
        graphs_na = loadGraphs(f"{folder}/no_anomaly/{n}/transform/Graph/WeightedGraph_2.pickle")
        graphs_a = loadGraphs(f"{folder}/anomaly/{n}/transform/Graph/WeightedGraph_2.pickle")

        graphs = graphs_na + graphs_a
        center_index = len(graphs_na)
        labels = label_snapshots(len(graphs), center_index, anomaly_window_size)

        d[n] = {
            "node": node,
            "graphs": graphs,
            "labels": labels
        }

    return d

if __name__ == "__main__":
    print("Transforming data to PyTorch Geometric")
    print("It may take several minutes, please wait ...")

    data = loadData("data/", anomaly_window_size=30)
    filename = "data_pyg.pickle"
    pickle.dump(data, open(filename, "wb"))

    print("Data saved to:", filename)
