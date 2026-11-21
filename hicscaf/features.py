"""Turn (components, Hi-C contacts, draft scaffolds) into model inputs.

Graph:  node = component (a contig or a piece of a contig that the draft uses)
        edge = candidate pair of components; label = one of 5 classes
               0 no link, 1 head-head, 2 head-tail, 3 tail-head, 4 tail-tail
               (end of A touching end of B, A = lower node index)
CNN:    for every junction in the draft, a 2-channel contact-map patch centred on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Component, Contacts, Scaffold, component_truth

HEAD, TAIL = 0, 1
SCALES = (20_000, 100_000, 500_000)


def link_class(end_a: int, end_b: int) -> int:
    return 1 + 2 * end_a + end_b


def decode_class(k: int) -> tuple[int, int]:
    k -= 1
    return k // 2, k % 2


# ------------------------------------------------------------ components
@dataclass
class ComponentSet:
    comps: list[Component]
    index: dict[Component, int]

    @classmethod
    def from_scaffolds(cls, scafs: list[Scaffold]):
        comps = sorted({c for s in scafs for c, _ in s.parts}, key=lambda c: (c.contig, c.beg))
        return cls(comps, {c: i for i, c in enumerate(comps)})

    @property
    def lengths(self) -> np.ndarray:
        return np.array([c.length for c in self.comps], np.int64)


def map_contacts(cs: ComponentSet, ct: Contacts):
    """Contig coordinates -> (component index, position inside component). Unmapped -> -1."""
    name_id = {n: i for i, n in enumerate(ct.names)}
    keys = np.array([name_id.get(c.contig, -1) * (1 << 40) + c.beg for c in cs.comps], np.int64)
    order = np.argsort(keys)
    keys_s = keys[order]
    ends = np.array([c.end for c in cs.comps], np.int64)
    begs = np.array([c.beg for c in cs.comps], np.int64)
    out = []
    w = ct.w
    for c, p in ((ct.c1, ct.p1), (ct.c2, ct.p2)):
        k = c.astype(np.int64) * (1 << 40) + p
        j = np.searchsorted(keys_s, k, side="right") - 1
        j = np.clip(j, 0, len(keys_s) - 1)
        idx = order[j]
        ok = (keys_s[j] >> 40 == c) & (p >= begs[idx]) & (p < ends[idx])
        idx = np.where(ok, idx, -1)
        out.append((idx, np.where(ok, p - begs[idx], 0)))
    (a, pa), (b, pb) = out
    m = (a >= 0) & (b >= 0)
    return a[m], pa[m], b[m], pb[m], w[m].astype(np.float64)


# --------------------------------------------------------- adjacency sets
def adjacencies_from_scaffolds(cs: ComponentSet, scafs: list[Scaffold]) -> dict[tuple[int, int], int]:
    out = {}
    for s in scafs:
        for (c1, s1), (c2, s2) in zip(s.parts, s.parts[1:]):
            a, b = cs.index[c1], cs.index[c2]
            ea = TAIL if s1 > 0 else HEAD
            eb = HEAD if s2 > 0 else TAIL
            if a > b:
                a, b, ea, eb = b, a, eb, ea
            out[(a, b)] = link_class(ea, eb)
    return out


def adjacencies_from_truth(cs: ComponentSet, truth) -> dict[tuple[int, int], int]:
    placed = []
    for i, c in enumerate(cs.comps):
        t = component_truth(c, truth)
        if t is not None:
            placed.append((t[0], t[1], t[2], t[3], i))
    placed.sort()
    out = {}
    for x, y in zip(placed, placed[1:]):
        if x[0] != y[0]:
            continue
        a, b = x[4], y[4]
        ea = TAIL if x[3] > 0 else HEAD
        eb = HEAD if y[3] > 0 else TAIL
        if a > b:
            a, b, ea, eb = b, a, eb, ea
        out[(a, b)] = link_class(ea, eb)
    return out


# --------------------------------------------------------------- features
@dataclass
class Graph:
    node_x: np.ndarray      # (N, Fn)
    edge_index: np.ndarray  # (E, 2) a < b
    edge_x: np.ndarray      # (E, Fe)
    edge_y: np.ndarray      # (E,) class or -1 if unknown
    lengths: np.ndarray


def _end_bin(pos, L, E):
    """0 head / 1 tail / -1 middle for an end-window size E (clipped to half the component)."""
    e = np.minimum(E, L // 2)
    return np.where(pos < e, HEAD, np.where(pos >= L - e, TAIL, -1))


def build_graph(cs: ComponentSet, mapped, draft_adj: dict, junction_score: dict | None = None,
                true_adj: dict | None = None, top_k: int = 10) -> Graph:
    a, pa, b, pb, w = mapped
    L = cs.lengths
    N = len(L)
    inter = a != b
    intra_cnt = np.bincount(a[~inter], weights=w[~inter], minlength=N) * 2
    ai, bi, pai, pbi, wi = a[inter], b[inter], pa[inter], pb[inter], w[inter]
    # canonical a < b
    sw = ai > bi
    ai, bi = np.where(sw, bi, ai), np.where(sw, ai, bi)
    pai, pbi = np.where(sw, pbi, pai), np.where(sw, pai, pbi)
    inter_cnt = np.bincount(ai, weights=wi, minlength=N) + np.bincount(bi, weights=wi, minlength=N)
    cov = (intra_cnt + inter_cnt) / np.maximum(L, 1) * 1e4  # contacts per 10 kb

    # --- candidate edges: top-k partners by length-normalised contacts + all draft adjacencies
    key = ai.astype(np.int64) * N + bi
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv, weights=wi, minlength=len(uk))
    ea_, eb_ = uk // N, uk % N
    score = cnt / np.sqrt(np.minimum(L[ea_], 1_000_000) * np.minimum(L[eb_], 1_000_000))
    cand = set()
    # vectorised top-k per node
    for side in (ea_, eb_):
        order = np.lexsort((-score, side))
        s_sorted = side[order]
        first = np.searchsorted(s_sorted, s_sorted, side="left")
        rank = np.arange(len(order)) - first
        keep = order[rank < top_k]
        cand.update(zip(ea_[keep].tolist(), eb_[keep].tolist()))
    cand.update(draft_adj.keys())
    edges = np.array(sorted(cand), np.int64).reshape(-1, 2)
    E = len(edges)
    eidx = {(int(x), int(y)): i for i, (x, y) in enumerate(edges)}

    # --- end-contact counts at multiple scales
    feats = []
    ekey = edges[:, 0] * N + edges[:, 1]
    pos_in = np.searchsorted(ekey, key)
    pos_in = np.clip(pos_in, 0, max(E - 1, 0))
    valid = (E > 0) & (ekey[pos_in] == key) if E else np.zeros(len(key), bool)
    for S in SCALES:
        ea_end = _end_bin(pai, L[ai], S)
        eb_end = _end_bin(pbi, L[bi], S)
        m = valid & (ea_end >= 0) & (eb_end >= 0)
        cls = 2 * ea_end[m] + eb_end[m]
        C = np.zeros((E, 4))
        np.add.at(C, (pos_in[m], cls), wi[m])
        # totals of each node end towards any other node (for "is this the best partner" ratios)
        endtot = np.zeros((N, 2))
        mm = (ea_end >= 0)
        np.add.at(endtot, (ai[mm], ea_end[mm]), wi[mm])
        mm = (eb_end >= 0)
        np.add.at(endtot, (bi[mm], eb_end[mm]), wi[mm])
        ends_a = np.array([HEAD, HEAD, TAIL, TAIL])
        ends_b = np.array([HEAD, TAIL, HEAD, TAIL])
        tot_a = endtot[edges[:, 0]][:, ends_a]
        tot_b = endtot[edges[:, 1]][:, ends_b]
        # expected under coverage: proportional to end sizes and coverage
        sa = np.minimum(S, L[edges[:, 0]] // 2)[:, None]
        sb = np.minimum(S, L[edges[:, 1]] // 2)[:, None]
        feats += [np.log1p(C), C / (C.sum(1, keepdims=True) + 1), C / (tot_a + 1), C / (tot_b + 1),
                  np.log1p(C / (np.sqrt(sa * sb) / 1e4 * np.sqrt(cov[edges[:, 0]] * cov[edges[:, 1]])[:, None] + 1e-3))]
    tot = np.zeros(E)
    if E:
        np.add.at(tot, pos_in[valid], wi[valid])
    prior = np.zeros((E, 5))
    jsc = np.full((E, 1), 0.5)
    has_j = np.zeros((E, 1))
    for (x, y), k in draft_adj.items():
        i = eidx[(x, y)]
        prior[i, k] = 1
        if junction_score is not None and (x, y) in junction_score:
            jsc[i, 0] = junction_score[(x, y)]
            has_j[i, 0] = 1
    prior[prior.sum(1) == 0, 0] = 1
    edge_x = np.concatenate(feats + [np.log1p(tot)[:, None], prior, jsc, has_j,
                                     np.log10(L[edges[:, 0]] + 1)[:, None] / 7,
                                     np.log10(L[edges[:, 1]] + 1)[:, None] / 7], axis=1).astype(np.float32)
    node_x = np.stack([np.log10(L + 1) / 7, np.log1p(cov), intra_cnt / (intra_cnt + inter_cnt + 1),
                       np.log1p(inter_cnt)], axis=1).astype(np.float32)
    y = np.full(E, -1 if true_adj is None else 0, np.int64)
    if true_adj is not None:
        for k, v in true_adj.items():
            if k in eidx:
                y[eidx[k]] = v
    return Graph(node_x, edges, edge_x, y, L)


N_EDGE_FEATS = len(SCALES) * 20 + 1 + 5 + 2 + 2
N_NODE_FEATS = 4


# ------------------------------------------------------------ CNN patches
def auto_bin_size(mapped, total_len: int, min_bin: int = 1_000) -> int:
    n = float(mapped[4].sum())
    ppm = n / max(total_len, 1) * 1e6
    return int(np.clip(5_000 * np.sqrt(40_000 / max(ppm, 1)), min_bin, 100_000) // 500 * 500)


def junction_patches(cs: ComponentSet, scafs: list[Scaffold], mapped, bin_size: int, half: int = 24):
    """Return patches (J, 2, 2h, 2h), list of junction keys (a,b) and the draft class for each."""
    a, pa, b, pb, w = mapped
    N = len(cs.comps)
    L = cs.lengths
    scaf_of = np.full(N, -1)
    off = np.zeros(N, np.int64)
    strand = np.ones(N, np.int64)
    for si, s in enumerate(scafs):
        pos = 0
        for c, st in s.parts:
            i = cs.index[c]
            scaf_of[i], off[i], strand[i] = si, pos, st
            pos += c.length
    sa, sb = scaf_of[a], scaf_of[b]
    same = (sa == sb) & (sa >= 0)
    x = off[a] + np.where(strand[a] > 0, pa, L[a] - 1 - pa)
    y = off[b] + np.where(strand[b] > 0, pb, L[b] - 1 - pb)
    sid, x, y, ww = sa[same], x[same], y[same], w[same]
    # symmetric
    sid = np.concatenate([sid, sid]); x, y = np.concatenate([x, y]), np.concatenate([y, x])
    ww = np.concatenate([ww, ww])
    order = np.lexsort((x, sid))
    sid, x, y, ww = sid[order], x[order], y[order], ww[order]
    keys, patches, classes = [], [], []
    W = 2 * half
    for si, s in enumerate(scafs):
        lo_s, hi_s = np.searchsorted(sid, si, "left"), np.searchsorted(sid, si, "right")
        xs, ys, wss = x[lo_s:hi_s], y[lo_s:hi_s], ww[lo_s:hi_s]
        pos = 0
        for (c1, s1), (c2, s2) in zip(s.parts, s.parts[1:]):
            pos += c1.length
            lo = pos - half * bin_size
            i0, i1 = np.searchsorted(xs, lo), np.searchsorted(xs, lo + W * bin_size)
            bx = (xs[i0:i1] - lo) // bin_size
            by = (ys[i0:i1] - lo) // bin_size
            m = (by >= 0) & (by < W)
            H = np.zeros((W, W), np.float32)
            np.add.at(H, (bx[m], by[m]), wss[i0:i1][m])
            patches.append(_normalise_patch(H))
            ia, ib = cs.index[c1], cs.index[c2]
            ea = TAIL if s1 > 0 else HEAD
            eb = HEAD if s2 > 0 else TAIL
            if ia > ib:
                ia, ib, ea, eb = ib, ia, eb, ea
            keys.append((ia, ib))
            classes.append(link_class(ea, eb))
    P = np.stack(patches) if patches else np.zeros((0, 2, W, W), np.float32)
    return P, keys, classes


def _normalise_patch(H: np.ndarray) -> np.ndarray:
    W = H.shape[0]
    i, j = np.indices(H.shape)
    d = np.abs(i - j)
    exp = np.bincount(d.ravel(), weights=H.ravel(), minlength=W) / np.bincount(d.ravel(), minlength=W)
    oe = np.log2((H + 0.5) / (exp[d] + 0.5))
    lg = np.log1p(H)
    lg = lg / (lg.max() + 1e-6)
    return np.stack([lg, np.clip(oe, -4, 4) / 4]).astype(np.float32)
