# -*- coding: utf-8 -*-
"""
BGNN 평가 (leave-one-event-out, demo 학습 방식 + 안정화 옵션)
----------------------------------------------------------------
* 모델  : BGP_GNN(nbIp+degree, logit 출력)
* 손실  : BCEWithLogitsLoss(pos_weight)
* 라벨  : window 마지막 index ≥ 44 → 1
* 옵티마: AdamW(lr=3e-4, wd=1e-4) + CosineAnnealingLR
* grad clip: 2.0
* run_loss : 지수평활(0.9)로 출력
"""
import csv, os, pickle, random, numpy as np, torch
from typing import List, Tuple
from tqdm.auto import tqdm
from model import BGP_GNN
from collections import defaultdict

def get_windows(graphs: List, labels: List[int], w: int):
    return [(graphs[i:i+w], 1 if i+w-1 >= 44 else 0)
            for i in range(len(graphs)-w+1)]

def _step(model, g, n, y, device, opt, crit):
    logit  = model(g, n)
    target = torch.tensor(y, dtype=torch.float, device=device)
    loss   = crit(logit, target)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)   
    opt.step()
    return loss.item()

def train(data, train_evts, seed, epochs, lr,
          layer, w, batch_size, device,
          hidden_dim=64, heads=4, dropout=0.2):

    torch.manual_seed(seed); random.seed(seed); 

    model = BGP_GNN(layer=layer, hidden_dim=hidden_dim,
                    heads=heads, dropout=dropout,
                    device=device).to(device)

    # pos_weight 계산
    pos = neg = 0
    for ev in train_evts:
        for _, y in get_windows(data[ev]["graphs"], data[ev]["labels"], w):
            pos += y; neg += 1 - y
    pos_w  = torch.tensor([neg/max(pos,1)], device=device)
    crit   = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)  
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)     

    for ep in range(1, epochs+1):
        model.train(); run_loss = 0.0
        buf_g, buf_n, buf_y = [], [], []

        pbar = tqdm(train_evts, desc=f"[S{seed}] Epoch {ep}", leave=False)
        for ev in pbar:
            node   = data[ev]["node"]
            graphs = data[ev]["graphs"]; labels = data[ev]["labels"]

            for g_seq, y in get_windows(graphs, labels, w):
                buf_g.append(g_seq); buf_n.append(node); buf_y.append([y])
                if len(buf_g) == batch_size:
                    loss = _step(model, buf_g, buf_n, buf_y,
                                 device, opt, crit)
                    run_loss = run_loss*0.9 + loss*0.1 
                    buf_g, buf_n, buf_y = [], [], []
            
            if buf_g:
                loss = _step(model, buf_g, buf_n, buf_y,
                             device, opt, crit)
                run_loss = run_loss*0.9 + loss*0.1
                buf_g, buf_n, buf_y = [], [], []

        print(f"  ▸ Epoch {ep}: run_loss={run_loss:.4f} "
              f"lr={sch.get_last_lr()[0]:.2e}")
        sch.step()                                                     

    return model

@torch.no_grad()
def test(data, ev, model, w, device):
    model.eval()
    node   = data[ev]["node"]
    graphs = data[ev]["graphs"]; labels = data[ev]["labels"]
    y_t, y_s, y_p = [], [], []
    for g_seq, y in get_windows(graphs, labels, w):
        s = torch.sigmoid(model([g_seq], [node])).item()
        y_t.append(y); y_s.append(s); y_p.append(int(s>0.5))
    return y_t, y_s, y_p

# ───────────────────────── main ───────────────────────────
if __name__ == "__main__":
    device      = "cuda"  # if torch.cuda.is_available() else "cpu"
    nb_seeds    = 5
    nb_epochs   = 10
    window_size = 10
    batch_size  = 1
    layers      = [1]

    data   = pickle.load(open("data_pyg.pickle", "rb"))
    events = list(data.keys())

    results = {}
    window_acc_dict = defaultdict(list)        
    prog = tqdm(total=len(layers)*nb_seeds*len(events), desc="Overall")

    for layer in layers:
        results[layer] = {}
        for seed in range(nb_seeds):
            res = {"y_true": [], "y_score": [], "y_pred": [],
                   "model_state_dict": {}}

            for test_ev in events:
                train_evts = [e for e in events if e != test_ev]

                model = train(data, train_evts, seed,
                              nb_epochs, 1e-3, layer,
                              window_size, batch_size, device)

                y_t, y_s, y_p = test(data, test_ev, model,
                                     window_size, device)

                for idx, (gt, pred) in enumerate(zip(y_t, y_p)):
                    if gt == 1:
                        window_acc_dict[idx].append(int(gt == pred))

                res["y_true"]  += y_t
                res["y_score"] += y_s
                res["y_pred"]  += y_p
                res["model_state_dict"][test_ev] = model.state_dict()
                prog.update(1)

            results[layer][seed] = res

    # ── 기존 결과 저장 ────────────────────────────────────
    with open("results.pickle", "wb") as f:
        pickle.dump(results, f)
    print("✅ results.pickle saved")

    # ── window-level accuracy 저장 ───────────────────────
    with open("window_accuracy.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["window_idx", "accuracy", "support"])
        for idx in sorted(window_acc_dict.keys()):
            flags   = window_acc_dict[idx]
            acc     = sum(flags) / len(flags)
            wr.writerow([idx, f"{acc:.4f}", len(flags)])
    print("✅ window_accuracy.csv saved")