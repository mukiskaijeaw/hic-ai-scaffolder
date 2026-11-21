"""Inference: draft scaffolds + Hi-C -> corrected scaffolds."""
from __future__ import annotations

import numpy as np
import torch

from .features import (HEAD, TAIL, ComponentSet, adjacencies_from_scaffolds, auto_bin_size, build_graph,
                       decode_class, junction_patches, map_contacts)
from .io import Contacts, Scaffold, add_missing_components


@torch.no_grad()
def score_junctions(cnn, patches: np.ndarray, batch: int = 512) -> np.ndarray:
    if cnn is None or len(patches) == 0:
        return np.full(len(patches), 0.5)
    out = []
    for i in range(0, len(patches), batch):
        out.append(torch.sigmoid(cnn(torch.from_numpy(patches[i : i + batch]))).numpy())
    return np.concatenate(out)


def prepare(lengths: dict[str, int], contacts: Contacts, draft: list[Scaffold] | None, cnn=None,
            bin_size: int | None = None, true_adj_fn=None):
    """Shared by training and inference. Returns (cs, graph, patches, jkeys, jclasses, bin_size)."""
    draft = add_missing_components(draft or [], lengths)
    cs = ComponentSet.from_scaffolds(draft)
    mapped = map_contacts(cs, contacts)
    bin_size = bin_size or auto_bin_size(mapped, int(cs.lengths.sum()), min_bin=max(1_000, 2 * contacts.resolution))
    dadj = adjacencies_from_scaffolds(cs, draft)
    patches, jkeys, jcls = junction_patches(cs, draft, mapped, bin_size)
    jscore = dict(zip(jkeys, score_junctions(cnn, patches)))
    true_adj = true_adj_fn(cs) if true_adj_fn else None
    g = build_graph(cs, mapped, dadj, jscore, true_adj)
    return cs, g, patches, jkeys, jcls, bin_size, draft


@torch.no_grad()
def edge_probs(gnn, g) -> np.ndarray:
    logits = gnn(torch.from_numpy(g.node_x), torch.from_numpy(g.edge_index), torch.from_numpy(g.edge_x))
    return torch.softmax(logits, -1).numpy()


def greedy_decode(cs: ComponentSet, edges: np.ndarray, probs: np.ndarray, min_prob: float = 0.5,
                  prefix: str = "scaffold_") -> list[Scaffold]:
    """Accept links best-first; each contig end used once; no cycles (union-find)."""
    N = len(cs.comps)
    cand = []
    for i, (a, b) in enumerate(edges):
        for k in range(1, 5):
            if probs[i, k] >= min_prob:
                cand.append((probs[i, k], int(a), int(b), k))
    cand.sort(reverse=True)
    partner = {}  # (node, end) -> (node, end)
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p, a, b, k in cand:
        ea, eb = decode_class(k)
        if (a, ea) in partner or (b, eb) in partner:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[ra] = rb
        partner[(a, ea)] = (b, eb)
        partner[(b, eb)] = (a, ea)

    seen = [False] * N
    scafs = []
    starts = [n for n in range(N) if (n, HEAD) not in partner or (n, TAIL) not in partner]
    for s in starts:
        if seen[s]:
            continue
        left_end = HEAD if (s, HEAD) not in partner else TAIL
        node, strand = s, (1 if left_end == HEAD else -1)
        parts = []
        while True:
            seen[node] = True
            parts.append((cs.comps[node], strand))
            right = TAIL if strand > 0 else HEAD
            nxt = partner.get((node, right))
            if nxt is None:
                break
            node, e = nxt
            strand = 1 if e == HEAD else -1
        scafs.append(parts)
    scafs.sort(key=lambda p: -sum(c.length for c, _ in p))
    return [Scaffold(f"{prefix}{i + 1}", p) for i, p in enumerate(scafs)]


def correct(lengths, contacts, draft, cnn, gnn, min_prob=0.5, bin_size=None, n_chrom: int = 0):
    """n_chrom > 0: afterwards group the scaffolds into that many chromosomes (see grouping.py)."""
    cs, g, *_ = prepare(lengths, contacts, draft, cnn, bin_size)
    probs = edge_probs(gnn, g)
    scafs = greedy_decode(cs, g.edge_index, probs, min_prob)
    if n_chrom:
        from .features import map_contacts
        from .grouping import group_chromosomes
        scafs = group_chromosomes(cs, map_contacts(cs, contacts), scafs, n_chrom)
    return scafs


def heuristic_probs(g) -> np.ndarray:
    """Model-free baseline (SALSA/YaHS-style reciprocal-best end links).

    Uses the 100 kb end-contact ratios: link strength = min(share of A-end contacts, share of B-end contacts).
    """
    s = 1  # scale index of 100 kb in SCALES
    base = s * 20
    ra = g.edge_x[:, base + 8 : base + 12]
    rb = g.edge_x[:, base + 12 : base + 16]
    p = np.minimum(ra, rb)
    return np.concatenate([1 - p.max(1, keepdims=True), p], 1)


def heuristic_scaffold(lengths, contacts, min_score=0.1):
    cs, g, *_ = prepare(lengths, contacts, None, None)
    return greedy_decode(cs, g.edge_index, heuristic_probs(g), min_score)
