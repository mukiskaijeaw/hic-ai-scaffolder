"""Neural networks: JunctionCNN (misjoin detector) and EdgeGNN (adjacency + orientation)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JunctionCNN(nn.Module):
    """Input (B, 2, 48, 48) contact patch centred on a draft junction -> P(junction is correct)."""

    def __init__(self, ch: int = 32):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.body = nn.Sequential(block(2, ch), block(ch, ch * 2), block(ch * 2, ch * 4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(ch * 4, 1))

    def forward(self, x):
        # the matrix is symmetric: average the prediction over the transpose
        return 0.5 * (self.head(self.body(x)) + self.head(self.body(x.transpose(-1, -2)))).squeeze(-1)


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


# swapping A and B maps class (ea, eb) -> (eb, ea): 1 HH->HH, 2 HT->TH, 3 TH->HT, 4 TT->TT
SWAP_PERM = [0, 1, 3, 2, 4]


class EdgeGNN(nn.Module):
    """Message passing on the contig graph; edge logits over 5 link classes.

    Each layer:  e <- e + MLP([h_a, h_b, e]);   h <- h + MLP([h, mean_e(incident), max_e(incident)])
    Node context lets an edge know whether its ends already have a better partner,
    which is exactly the information a greedy scaffolder lacks.
    """

    def __init__(self, n_node: int, n_edge: int, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.node_in = mlp(n_node, hidden, hidden)
        self.edge_in = mlp(n_edge, hidden, hidden)
        self.edge_up = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(layers)])
        self.node_up = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(layers)])
        self.out = mlp(3 * hidden, hidden, 5)

    def forward(self, node_x, edge_index, edge_x):
        h = self.node_in(node_x)
        e = self.edge_in(edge_x)
        a, b = edge_index[:, 0], edge_index[:, 1]
        N = h.shape[0]
        for eu, nu in zip(self.edge_up, self.node_up):
            e = e + eu(torch.cat([h[a] + h[b], (h[a] - h[b]).abs(), e], -1))
            idx = torch.cat([a, b])
            msg = torch.cat([e, e])
            s = torch.zeros(N, e.shape[1], device=e.device).index_add_(0, idx, msg)
            deg = torch.zeros(N, device=e.device).index_add_(0, idx, torch.ones_like(idx, dtype=e.dtype)).clamp(min=1)
            mx = torch.full((N, e.shape[1]), -1e4, device=e.device).scatter_reduce(
                0, idx[:, None].expand_as(msg), msg, reduce="amax", include_self=True)
            mx = torch.where(mx < -1e3, torch.zeros_like(mx), mx)
            h = h + nu(torch.cat([h, s / deg[:, None], mx], -1))
        return self.out(torch.cat([h[a], h[b], e], -1))


def save(path, cnn: JunctionCNN | None, gnn: EdgeGNN | None, meta: dict):
    torch.save({"cnn": cnn.state_dict() if cnn else None, "gnn": gnn.state_dict() if gnn else None, "meta": meta}, path)


def load(path, device="cpu"):
    from .features import N_EDGE_FEATS, N_NODE_FEATS

    ck = torch.load(path, map_location=device, weights_only=False)
    meta = ck["meta"]
    cnn = gnn = None
    if ck["cnn"] is not None:
        cnn = JunctionCNN(meta.get("cnn_ch", 32))
        cnn.load_state_dict(ck["cnn"]); cnn.eval()
    if ck["gnn"] is not None:
        gnn = EdgeGNN(N_NODE_FEATS, N_EDGE_FEATS, meta.get("hidden", 128), meta.get("layers", 4))
        gnn.load_state_dict(ck["gnn"]); gnn.eval()
    return cnn, gnn, meta
