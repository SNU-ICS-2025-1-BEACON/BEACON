# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch
from torch_geometric.utils import degree


class BGP_GNN(nn.Module):
    """
    Node-level feature : [ nbIp , degree ]
    Snapshot feature   : snap_feat  (MLP → hidden_dim)
    GAT → GRU → MLP → **logit**   (sigmoid X)
    """
    def __init__(self,
                 layer: int = 1,
                 hidden_dim: int = 64,
                 heads: int = 4,
                 dropout: float = 0.2,
                 device: str = "cpu"):
        super().__init__()
        self.device      = device
        self.hidden_dim  = hidden_dim
        self.layer       = layer
        self.heads       = heads
        self.dropout_p   = dropout

        # GAT 스택 (in_channels = 2  : nbIp + degree)
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels = 2 if i == 0 else hidden_dim,
                out_channels = hidden_dim // heads,
                heads=heads,
                concat=True,          # out × heads = hidden_dim
                dropout=dropout
            ).to(device)
            for i in range(layer)
        ])

        # snapshot-feature MLP (lazy, 크기는 forward에서 초기화)
        self.feat_mlp: nn.Module | None = None

        # Temporal + Classifier
        self.gru = nn.GRU(hidden_dim, hidden_dim,
                          batch_first=True,
                          dropout=0. if layer == 1 else dropout)
        self.linear1 = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.linear2 = nn.Linear(32, 1)     # ★ sigmoid 없음 (logit 반환)

    # ----------------------------------------------------------
    def forward(self,
                batch_graph_seqs: list[list],
                batch_nodes: list[str | int]) -> torch.Tensor:

        # ── flatten ───────────────────────────────────────────
        graphs_flat, seq_lens = [], []
        for seq in batch_graph_seqs:
            graphs_flat += seq
            seq_lens.append(len(seq))

        batch = Batch.from_data_list(graphs_flat).to(self.device)

        # ── node-feature (nbIp + degree) ─────────────────────
        if not hasattr(batch, "nbIp"):
            raise RuntimeError("Data 객체에 nbIp 텐서가 없습니다.")

        nbip = batch.nbIp.view(-1, 1).float()
        deg  = degree(batch.edge_index[0],
                      num_nodes=batch.num_nodes).view(-1, 1).float()
        x = torch.cat([nbip, deg], dim=1)                 # (N,2)

        # ── GAT ──────────────────────────────────────────────
        for gat in self.gat_layers:
            x = gat(x, batch.edge_index).relu()
        x = nn.functional.dropout(x, p=self.dropout_p, training=self.training)

        # ── snapshot-feature MLP lazy init ───────────────────
        if self.feat_mlp is None:
            feat_dim = (
                graphs_flat[0].snap_feat.numel()
                if hasattr(graphs_flat[0], "snap_feat") else 0
            )
            self.feat_mlp = (
                nn.Identity() if feat_dim == 0 else
                nn.Sequential(
                    nn.Linear(feat_dim, self.hidden_dim),
                    nn.ReLU(), nn.LayerNorm(self.hidden_dim)
                ).to(self.device)
            )

        # ── 시퀀스별 타깃 노드 임베딩 + fusion ────────────────
        ptr, node_emb_list, g_idx = batch.ptr.tolist(), [], 0
        for b, T in enumerate(seq_lens):
            idx = []
            for t in range(T):
                g = batch_graph_seqs[b][t]
                idx.append(ptr[g_idx] + g.ASN.index(batch_nodes[b]))
                g_idx += 1
            h_node = x[idx]                                 # (T,H)

            if hasattr(batch_graph_seqs[b][0], "snap_feat"):
                snap = torch.stack(
                    [g.snap_feat for g in batch_graph_seqs[b]], dim=0
                ).to(self.device)                           # (T,F)
                h_feat = self.feat_mlp(snap)                # (T,H)
            else:
                h_feat = torch.zeros_like(h_node)

            h_fuse = nn.functional.layer_norm(h_node + h_feat,
                                               (self.hidden_dim,))
            node_emb_list.append(h_fuse)

        node_emb = torch.stack(node_emb_list, dim=0)        # (B,T,H)

        # ── GRU + classifier ────────────────────────────────
        _, h = self.gru(node_emb)                           # (1,B,H)
        out = self.linear1(h.squeeze(0))
        logit = self.linear2(out)                           # (B,1)
        return logit
