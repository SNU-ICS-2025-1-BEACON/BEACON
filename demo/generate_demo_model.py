# -*- coding: utf-8 -*-
"""
Demo BGNN 학습 ― eval.py 와 동일한 학습 파이프라인
────────────────────────────────────────────────────────
* AdamW(lr=3e-4, wd=1e-4)  +  CosineAnnealingLR
* grad clip 2.0
* run_loss = 지수평활(0.9)
* pos_weight 적용 BCEWithLogitsLoss
"""
import csv, pickle, random, time, numpy as np, torch
from collections import defaultdict
from tqdm import tqdm
from model import BGP_GNN

def get_windows(graphs, labels, w):
    return [(graphs[i:i+w], 1 if i+w-1 >= 44 else 0)
            for i in range(len(graphs)-w+1)]

def _step(model, g, n, y, device, opt, crit):
    logit  = model(g, n)
    target = torch.tensor(y, dtype=torch.float, device=device)
    loss   = crit(logit, target)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)   # grad-clip
    opt.step()
    return loss.item()

# ───── 설정 ─────────────────────────────────────────────
device      = "cuda" #if torch.cuda.is_available() else "cpu"
nb_epoch    = 10
nb_seed     = 1
layer       = 1
w           = 10
batch_size  = 1
time_log    = "training_times.csv"

data   = pickle.load(open("../data_pyg.pickle", "rb"))
events = list(data.keys())
test   = events.pop(0)        # 첫 이벤트만 테스트로 제외

for seed in range(nb_seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    model = BGP_GNN(layer=layer, device=device).to(device)

    # pos_weight 계산 (train events 만)
    pos = neg = 0
    for ev in events:
        for _, y in get_windows(data[ev]["graphs"], data[ev]["labels"], w):
            pos += y; neg += 1 - y
    pos_w = torch.tensor([neg/max(pos,1)], device=device)
    crit  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nb_epoch)

    print(f"[Seed {seed}]  pos={pos}  neg={neg}  "
          f"pos_weight={pos_w.item():.2f}")

    total_start = time.time()
    event_times = defaultdict(list)

    for ep in range(1, nb_epoch+1):
        model.train(); run_loss = 0.0
        buf_g, buf_n, buf_l = [], [], []

        for ev in tqdm(events, desc=f"Epoch {ep}/{nb_epoch}", leave=False):
            node   = data[ev]["node"]
            graphs = data[ev]["graphs"]; labels = data[ev]["labels"]
            # t0 = time.time()

            for g_seq, y in get_windows(graphs, labels, w):
                buf_g.append(g_seq); buf_n.append(node); buf_l.append([y])
                if len(buf_g) == batch_size:
                    loss = _step(model, buf_g, buf_n, buf_l,
                                 device, opt, crit)
                    run_loss = run_loss*0.9 + loss*0.1
                    buf_g, buf_n, buf_l = [], [], []
            # flush
            if buf_g:
                loss = _step(model, buf_g, buf_n, buf_l,
                             device, opt, crit)
                run_loss = run_loss*0.9 + loss*0.1
                buf_g, buf_n, buf_l = [], [], []

            # event_times[ev].append((time.time() - t0)*1000)

        print(f"Epoch {ep}: run_loss={run_loss:.4f} "
              f"lr={sch.get_last_lr()[0]:.2e}")
        sch.step()

    print("Training done, total {:.1f}s".format(time.time()-total_start))
    torch.save(model.state_dict(), "model_demo.pt")
    print("model_demo.pt saved")

    # # 이벤트별 평균 학습시간 저장
    # with open(time_log, "w", newline="") as f:
    #     wr = csv.writer(f); wr.writerow(["event", "avg_ms"])
    #     for ev, ts in event_times.items():
    #         wr.writerow([ev, round(sum(ts)/len(ts), 2)])
    # print("per-event 평균 학습시간 →", time_log)