# -*- coding: utf-8 -*- 
"""
BGNN 평가 (leave-one-event-out, Cosine LR + 안정화 옵션)
─────────────────────────────────────────────────────────
* 모델 : BGP_GNN(nbIp+degree, logit 출력)
* 손실 : BCEWithLogitsLoss(pos_weight)
* 라벨 : window 마지막 index ≥ 44 → 1
* 옵티마: AdamW(lr=3e-4, wd=1e-4) + CosineAnnealingLR
* grad clip = 2.0, run_loss 출력
* 추가   : 각 슬라이딩-윈도우 추론 지연시간 저장 → inference_latency.csv
"""
import csv, os, pickle, random, numpy as np, torch, time
from collections import defaultdict
from typing import List, Tuple
from tqdm.auto import tqdm
from model import BGP_GNN

# ───────────────── window 생성 ───────────────────────────
def get_windows(graphs: List, labels: List[int], w: int):
    return [(graphs[i:i+w], 1 if i+w-1 >= 44 else 0)
            for i in range(len(graphs)-w+1)]

# ───────────────── grad 한 step ──────────────────────────
def _step(model, g, n, y, device, opt, crit):
    logit  = model(g, n)
    target = torch.tensor(y, dtype=torch.float, device=device)
    loss   = crit(logit, target)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    opt.step()
    return loss.item()

# ───────────────── Train 함수 ────────────────────────────
def train(data, train_evts, seed, epochs, layer, w,
          batch_size, device, hidden_dim=64, heads=4, dropout=0.2):

    torch.manual_seed(seed); random.seed(seed)

    model = BGP_GNN(layer=layer, hidden_dim=hidden_dim,
                    heads=heads, dropout=dropout,
                    device=device).to(device)

    # pos_weight
    pos = neg = 0
    for ev in train_evts:
        for _, y in get_windows(data[ev]["graphs"], data[ev]["labels"], w):
            pos += y; neg += 1 - y
    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg/max(pos,1)], device=device)
    )

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(1, epochs+1):
        model.train(); run_loss = 0.0
        buf_g, buf_n, buf_y = [], [], []

        for ev in tqdm(train_evts, desc=f"[S{seed}] Epoch {ep}", leave=False):
            node   = data[ev]["node"]
            graphs = data[ev]["graphs"]; labels = data[ev]["labels"]

            for g_seq, y in get_windows(graphs, labels, w):
                buf_g.append(g_seq); buf_n.append(node); buf_y.append([y])
                if len(buf_g) == batch_size:
                    loss = _step(model, buf_g, buf_n, buf_y,
                                 device, opt, crit)
                    run_loss = run_loss*0.9 + loss*0.1
                    buf_g, buf_n, buf_y = [], [], []

            if buf_g:  # flush
                loss = _step(model, buf_g, buf_n, buf_y,
                             device, opt, crit)
                run_loss = run_loss*0.9 + loss*0.1
                buf_g, buf_n, buf_y = [], [], []

        print(f"  ▸ Epoch {ep}: run_loss={run_loss:.4f} "
              f"lr={sch.get_last_lr()[0]:.2e}")
        sch.step()

    return model

# ───────────────── Test (추론 시간 포함) ─────────────────
@torch.no_grad()
def test(data, ev, model, w, device, seed, layer,
         latency_records: list):
    model.eval()
    node   = data[ev]["node"]
    graphs = data[ev]["graphs"]; labels = data[ev]["labels"]

    y_t, y_s, y_p = [], [], []
    for idx, (g_seq, y) in enumerate(get_windows(graphs, labels, w)):
        torch.cuda.synchronize() if device.startswith("cuda") else None
        t0 = time.time()
        s  = torch.sigmoid(model([g_seq], [node])).item()
        torch.cuda.synchronize() if device.startswith("cuda") else None
        elapsed_ms = (time.time() - t0) * 1000

        # latency 기록 ★
        latency_records.append({
            "seed": seed, "layer": layer, "event": ev,
            "window_idx": idx, "time_ms": round(elapsed_ms, 4)
        })

        y_t.append(y); y_s.append(s); y_p.append(int(s>0.5))
    return y_t, y_s, y_p

# ───────────────── main ──────────────────────────────────
if __name__ == "__main__":
    device       = "cuda" # if torch.cuda.is_available() else "cpu"
    nb_seeds     = 1
    nb_epochs    = 10
    window_size  = 10
    batch_size   = 1
    layers       = [1]

    data   = pickle.load(open("data_pyg.pickle", "rb"))
    events = list(data.keys())

    results = {}
    latency_log = []                    # ★  모든 window 추론 시간 저장
    prog = tqdm(total=len(layers)*nb_seeds*len(events), desc="Overall")

    for layer in layers:
        for seed in range(nb_seeds):
            for test_ev in events:
                train_evts = [e for e in events if e != test_ev]

                model = train(data, train_evts=train_evts, seed=seed,
                              epochs=nb_epochs, layer=layer,
                              w=window_size, batch_size=batch_size,
                              device=device)

                y_t, y_s, y_p = test(data, test_ev, model,
                                     window_size, device, seed, layer,
                                     latency_log)       # ★ latency 기록

                prog.update(1)

    # — 결과 파일 3: inference_latency.csv —
    with open("inference_latency.csv", "w", newline="") as f:
        wr = csv.DictWriter(f,
            fieldnames=["seed","layer","event","window_idx","time_ms"])
        wr.writeheader(); wr.writerows(latency_log)
    print("✅ inference_latency.csv saved")
