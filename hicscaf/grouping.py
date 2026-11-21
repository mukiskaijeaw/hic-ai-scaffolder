"""Group scaffolds into a fixed number of chromosomes, then order and orient each group.

Used when the user knows the karyotype (e.g. 2n = 2x = 24 -> 12 chromosomes), as with ALLHiC `-k`
or LACHESIS `CLUSTER_N`. Starts from the scaffolds the decoder already built (their joins are kept)
and adds the joins that make one scaffold per chromosome.

1. cluster: average-linkage agglomerative clustering on the length-normalised contact density
   between scaffolds (LACHESIS-style), using only scaffolds long enough to carry signal;
2. assign: the remaining short scaffolds join the group they contact most, when the signal is clear;
3. order + orient: inside each group, link scaffold ends greedily by end-weighted contacts
   (a contact counts more the closer it lies to the two ends being joined).
Scaffolds that cannot be placed confidently stay unplaced and are written after the chromosomes.
"""
from __future__ import annotations

import numpy as np

from .io import Scaffold


def _unit_coords(cs, scafs):
    """Per component: unit (scaffold) index, offset of the component in it, strand; plus unit lengths."""
    N = len(cs.comps)
    unit = np.full(N, -1, np.int64)
    off = np.zeros(N, np.int64)
    strand = np.ones(N, np.int64)
    ulen = np.zeros(len(scafs), np.int64)
    for u, s in enumerate(scafs):
        pos = 0
        for c, st in s.parts:
            i = cs.index[c]
            unit[i], off[i], strand[i] = u, pos, st
            pos += c.length
        ulen[u] = pos
    return unit, off, strand, ulen


def _cluster(M: np.ndarray, L: np.ndarray, k: int) -> np.ndarray:
    """Average-linkage on density M[i,j] / (L_i L_j); returns a cluster label per row (0..k-1)."""
    n = len(L)
    M = M.astype(np.float64).copy()
    np.fill_diagonal(M, 0)
    L = L.astype(np.float64).copy()
    alive = np.ones(n, bool)
    label = np.arange(n)

    def row(i):
        d = M[i] / (L[i] * L)
        d[~alive] = -1
        d[i] = -1
        return d

    best = np.empty(n, np.int64)
    bval = np.empty(n)
    for i in range(n):
        d = row(i)
        best[i] = int(np.argmax(d))
        bval[i] = d[best[i]]
    left = n
    while left > k:
        cand = np.where(alive, bval, -np.inf)
        i = int(np.argmax(cand))
        if cand[i] <= 0:  # nothing links any more: stop (leftovers handled by the caller)
            break
        j = int(best[i])
        # merge j into i
        M[i] += M[j]; M[:, i] += M[:, j]
        M[i, i] = 0
        M[j] = 0; M[:, j] = 0
        L[i] += L[j]
        alive[j] = False
        label[label == j] = i
        left -= 1
        d = row(i)
        best[i] = int(np.argmax(d)); bval[i] = d[best[i]]
        # rows whose best partner was i or j must be refreshed; others may now prefer i
        di = M[:, i] / (L * L[i])
        for r in np.where(alive)[0]:
            if r == i:
                continue
            if best[r] in (i, j):
                dr = row(r)
                best[r] = int(np.argmax(dr)); bval[r] = dr[best[r]]
            elif di[r] > bval[r]:
                best[r], bval[r] = i, di[r]
    roots = np.unique(label)
    # keep the k heaviest clusters (by length); the rest become "unassigned" (-1)
    size = np.array([L[r] for r in roots])
    keep = roots[np.argsort(-size)[:k]]
    remap = {r: t for t, r in enumerate(keep)}
    return np.array([remap.get(x, -1) for x in label])


def _order_group(members: list[int], ua, pa_u, ub, pb_u, w, ulen):
    """Order + orient units of one group. Returns [(unit, strand), ...]."""
    if len(members) == 1:
        return [(members[0], 1)]
    loc = {u: t for t, u in enumerate(members)}
    n = len(members)
    m = np.isin(ua, members) & np.isin(ub, members) & (ua != ub)
    a, b, pa, pb, ww = ua[m], ub[m], pa_u[m], pb_u[m], w[m]
    S = np.zeros((n, 2, n, 2))  # S[x, end_x, y, end_y]
    if len(a):
        ia = np.array([loc[x] for x in a]); ib = np.array([loc[x] for x in b])
        da = (pa, ulen[a] - 1 - pa)  # distance to head / tail
        db = (pb, ulen[b] - 1 - pb)
        for ea in (0, 1):
            for eb in (0, 1):
                s = ww / (da[ea] + db[eb] + 20_000.0)
                np.add.at(S, (ia, ea, ib, eb), s)
                np.add.at(S, (ib, eb, ia, ea), s)
    cand = [(S[x, ex, y, ey], x, ex, y, ey) for x in range(n) for y in range(x + 1, n)
            for ex in (0, 1) for ey in (0, 1) if S[x, ex, y, ey] > 0]
    cand.sort(reverse=True)
    partner: dict = {}
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, x, ex, y, ey in cand:
        if (x, ex) in partner or (y, ey) in partner:
            continue
        rx, ry = find(x), find(y)
        if rx == ry:
            continue
        parent[rx] = ry
        partner[(x, ex)] = (y, ey)
        partner[(y, ey)] = (x, ex)
    # walk paths
    seen = [False] * n
    paths = []
    for s in range(n):
        if seen[s] or ((s, 0) in partner and (s, 1) in partner):
            continue
        node, st = s, (1 if (s, 0) not in partner else -1)
        path = []
        while True:
            seen[node] = True
            path.append((members[node], st))
            nxt = partner.get((node, 1 if st > 0 else 0))
            if nxt is None:
                break
            node, e = nxt
            st = 1 if e == 0 else -1
        paths.append(path)
    paths.sort(key=lambda p: -sum(ulen[u] for u, _ in p))
    order = [x for p in paths for x in p]
    if len(a) and n > 2:
        order = _refine_order(order, a, pa, b, pb, ww, ulen)
    return order


def _refine_order(order, a, pa, b, pb, w, ulen, bin_bp: int = 25_000, window: int = 4, max_pass: int = 4,
                  budget_s: float = 20.0):
    """Local search on the order/orientation of a group's units.

    Minimises sum(w * log(d + 10 kb)) over inter-unit contacts, d = distance in the joined chromosome
    (the log-likelihood of a power-law contact decay). Moves: flip one unit, move one unit up to `window`
    places. Contacts are pooled into `bin_bp` bins inside each unit to keep each evaluation cheap.
    """
    import time

    members = [u for u, _ in order]
    loc = {u: t for t, u in enumerate(members)}
    n = len(members)
    ia = np.array([loc[x] for x in a]); ib = np.array([loc[x] for x in b])
    lens = np.array([ulen[u] for u in members], np.float64)
    nb = np.clip(np.ceil(lens / bin_bp), 4, 400)  # >= 4 bins so that flipping a short unit still matters
    qa = np.minimum((pa / lens[ia] * nb[ia]).astype(np.int64), nb[ia].astype(np.int64) - 1)
    qb = np.minimum((pb / lens[ib] * nb[ib]).astype(np.int64), nb[ib].astype(np.int64) - 1)
    key = ((ia * 1000 + qa) * n + ib) * 1000 + qb
    uk, inv = np.unique(key, return_inverse=True)
    W = np.bincount(inv, w)
    first = np.zeros(len(uk), np.int64); first[inv] = np.arange(len(inv))
    i1, i2 = ia[first], ib[first]
    c1 = (qa[first] + 0.5) * lens[i1] / nb[i1]
    c2 = (qb[first] + 0.5) * lens[i2] / nb[i2]

    def cost(perm, st):
        start = np.empty(n)
        start[perm] = np.concatenate([[0], np.cumsum(lens[perm])[:-1]])
        x = start[i1] + np.where(st[i1] > 0, c1, lens[i1] - c1)
        y = start[i2] + np.where(st[i2] > 0, c2, lens[i2] - c2)
        return float(np.dot(W, np.log(np.abs(x - y) + 10_000.0)))

    perm = np.arange(n)
    st = np.array([s for _, s in order])
    best = cost(perm, st)
    t0 = time.time()
    for _ in range(max_pass):
        improved = False
        for k in range(n):
            if time.time() - t0 > budget_s:
                break
            u = perm[k]
            st[u] = -st[u]
            c = cost(perm, st)
            if c < best - 1e-9:
                best, improved = c, True
            else:
                st[u] = -st[u]
            pos = int(np.where(perm == u)[0][0])
            rest = np.delete(perm, pos)
            bj, bc = None, best
            for j in range(max(0, pos - window), min(n - 1, pos + window) + 1):
                if j == pos:
                    continue
                cand = np.insert(rest, j, u)
                c = cost(cand, st)
                if c < bc - 1e-9:
                    bj, bc = cand, c
            if bj is not None:
                perm, best, improved = bj, bc, True
        if not improved:
            break
    return [(members[i], int(st[i])) for i in perm]


def group_chromosomes(cs, mapped, scafs: list[Scaffold], n_chrom: int, min_len: int = 50_000,
                      max_units: int = 3000, min_links: float = 3.0, min_share: float = 0.5,
                      prefix: str = "chr", progress=None) -> list[Scaffold]:
    """Join `scafs` (built on `cs`) into `n_chrom` chromosome scaffolds + unplaced leftovers."""
    if n_chrom <= 0 or not scafs:
        return scafs
    a, pa, b, pb, w = mapped
    unit, off, strand, ulen = _unit_coords(cs, scafs)
    L = cs.lengths
    ua, ub = unit[a], unit[b]
    pa_u = np.where(strand[a] > 0, off[a] + pa, off[a] + L[a] - 1 - pa)
    pb_u = np.where(strand[b] > 0, off[b] + pb, off[b] + L[b] - 1 - pb)
    U = len(scafs)
    inter = ua != ub
    links = np.bincount(ua[inter], w[inter], U) + np.bincount(ub[inter], w[inter], U)

    # 1 cluster the long, linked units
    elig = np.where((ulen >= min_len) & (links >= min_links))[0]
    if len(elig) > max_units:
        elig = elig[np.argsort(-ulen[elig])[:max_units]]
    if len(elig) < n_chrom:
        elig = np.argsort(-ulen)[:n_chrom]
    pos = np.full(U, -1, np.int64)
    pos[elig] = np.arange(len(elig))
    m = inter & (pos[ua] >= 0) & (pos[ub] >= 0)
    M = np.zeros((len(elig), len(elig)))
    np.add.at(M, (pos[ua[m]], pos[ub[m]]), w[m])
    M = M + M.T
    lab = _cluster(M, ulen[elig] / 1e6, n_chrom)
    group = np.full(U, -1, np.int64)
    group[elig] = lab
    if progress:
        progress(0.6)

    # 2 assign the rest by contacts to each group (density-normalised by group length)
    glen = np.bincount(group[group >= 0], ulen[group >= 0], n_chrom).astype(np.float64)
    for _ in range(2):  # second pass lets units link through newly assigned neighbours
        todo = group < 0
        ga, gb = group[ua], group[ub]
        sel1 = todo[ua] & (gb >= 0) & inter
        sel2 = todo[ub] & (ga >= 0) & inter
        C = np.zeros((U, n_chrom))
        np.add.at(C, (ua[sel1], gb[sel1]), w[sel1])
        np.add.at(C, (ub[sel2], ga[sel2]), w[sel2])
        tot = C.sum(1)
        D = C / np.maximum(glen, 1)
        bestg = D.argmax(1)
        share = C[np.arange(U), bestg] / np.maximum(tot, 1e-9)
        ok = todo & (tot >= min_links) & (share >= min_share)
        group[ok] = bestg[ok]
        glen = np.bincount(group[group >= 0], ulen[group >= 0], n_chrom).astype(np.float64)
    if progress:
        progress(0.75)

    # 3 order + orient inside each group
    chroms = []
    for g in range(n_chrom):
        members = [int(u) for u in np.where(group == g)[0]]
        if not members:
            continue
        order = _order_group(members, ua, pa_u, ub, pb_u, w, ulen)
        parts = []
        for u, st in order:
            p = scafs[u].parts
            parts += p if st > 0 else [(c, -s) for c, s in reversed(p)]
        chroms.append(parts)
    chroms.sort(key=lambda p: -sum(c.length for c, _ in p))
    out = [Scaffold(f"{prefix}{i + 1}", p) for i, p in enumerate(chroms)]
    rest = sorted((scafs[u] for u in np.where(group < 0)[0]), key=lambda s: -sum(c.length for c, _ in s.parts))
    out += [Scaffold(f"unplaced_{i + 1}", s.parts) for i, s in enumerate(rest)]
    if progress:
        progress(1.0)
    return out
