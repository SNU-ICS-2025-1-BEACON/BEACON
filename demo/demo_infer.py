# -*- coding: utf-8 -*-
import pickle
import time
import torch

device = "cuda" #if torch.cuda.is_available() else "cpu"
model_path = "model_demo.pt"
data_path = "../data_pyg.pickle"
w = 10
layer = 1

data = pickle.load(open(data_path, "rb"))
name = list(data.keys())[0]
print("Target Event: " + name)
node = data[name]["node"]
graphs = data[name]["graphs"]
labels = data[name]["labels"]

from model import BGP_GNN
    
def get_windows(graphs, labels, w, anomaly_ratio_threshold=0.5):
    windows = []
    for i in range(len(graphs) - w + 1):
        window_graphs = graphs[i:i+w]
        window_labels = labels[i:i+w]
        anomaly_ratio = sum(window_labels) / w
        label = 1 if i+w-1 >= 44 else 0
        windows.append((window_graphs, label))
    return windows


windows = get_windows(graphs, labels, w=w)

model = BGP_GNN(layer=layer, hidden_dim=64, heads=4, device=device).to(device)
model([windows[0][0]], [node])
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

from torch.nn.functional import sigmoid 

log_path = "demo_infer.log"
with open(log_path, "w") as log_file:
    with torch.no_grad():
        for i, (g_seq, label) in enumerate(windows):
            start = time.time()
            logit  = model([g_seq], [node]).item()
            score  = sigmoid(torch.tensor(logit)).item()
            elapsed = time.time() - start
            pred = int(score > 0.5)

            line = (f"[Window {i:02d}] "
                    f"Range={i}-{i+w-1} | "
                    f"Label: GT={label} | Pred={pred} | "
                    f"Score={score:.4f} | Time={elapsed:.6f}s")

            print(line)
            log_file.write(line + "\n")