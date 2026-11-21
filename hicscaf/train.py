"""Train JunctionCNN then EdgeGNN on synthetic genomes with free labels."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .features import N_EDGE_FEATS, N_NODE_FEATS, adjacencies_from_truth
from .models import EdgeGNN, JunctionCNN, save
from .scaffold import prepare, score_junctions
from .simulate import simulate


def bin_contacts(ct, lengths, res, rng):
    """Quantise positions to `res` bins with a random phase per contig (approximates matrix binning)."""
    from .io import Contacts
    L = np.array([lengths[n] for n in ct.names])
    phase = rng.integers(0, res, len(L))
    q = lambda c, p: np.clip((p + phase[c]) // res * res + res // 2 - phase[c], 0, L[c] - 1)
    return Contacts(ct.names, ct.c1, q(ct.c1, ct.p1), ct.c2, q(ct.c2, ct.p2), ct.w, res)


def make_examples(seeds, pairs_per_mb=None, chroms=None, log=print):
    """One example per synthetic genome. Contact depth is varied so the model sees shallow and deep data."""
    out = []
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(10_000 + s)
        ppm = pairs_per_mb or int(rng.choice([10_000, 20_000, 40_000, 80_000]))
        sim = simulate(seed=s, chroms=chroms, pairs_per_mb=ppm)
        # half of the genomes look like a binned .hic/.mcool matrix (refine from precomputed outputs)
        if rng.random() < 0.5:
            sim.contacts = bin_contacts(sim.contacts, sim.lengths, int(rng.choice([1000, 2000, 5000, 10000])), rng)
        cs, g, patches, jkeys, jcls, bin_size, draft = prepare(
            sim.lengths, sim.contacts, sim.draft, cnn=None, true_adj_fn=lambda cs: adjacencies_from_truth(cs, sim.truth))
        true = adjacencies_from_truth(cs, sim.truth)
        jy = np.array([float(true.get(k) == c) for k, c in zip(jkeys, jcls)], np.float32)
        out.append(dict(sim=sim, g=g, patches=patches, jy=jy, bin_size=bin_size))
        if (i + 1) % 10 == 0:
            log(f"  simulated {i + 1}/{len(seeds)} genomes")
    return out


def train_cnn(train, val, epochs=8, lr=2e-3, batch=128, log=print):
    X = np.concatenate([e["patches"] for e in train]); y = np.concatenate([e["jy"] for e in train])
    Xv = np.concatenate([e["patches"] for e in val]); yv = np.concatenate([e["jy"] for e in val])
    log(f"CNN: {len(X)} train junctions ({(1 - y.mean()) * 100:.1f}% wrong), {len(Xv)} val")
    model = JunctionCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    wrong_w = float(np.clip((y == 1).sum() / max((y == 0).sum(), 1), 1, 20))  # up-weight rare wrong junctions
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((len(X) + batch - 1) // batch))
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0
        for i in range(0, len(Xt), batch):
            idx = perm[i : i + batch]
            xb = Xt[idx]
            if np.random.rand() < 0.5:  # flipping the patch = reading the scaffold backwards
                xb = xb.flip(-1).flip(-2)
            logit = model(xb)
            w = torch.where(yt[idx] == 0, torch.tensor(wrong_w), torch.tensor(1.0))
            loss = F.binary_cross_entropy_with_logits(logit, yt[idx], weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        p = score_junctions(model, Xv)
        acc = ((p > 0.5) == (yv > 0.5)).mean()
        bad = yv == 0
        rec_bad = ((p < 0.5) & bad).sum() / max(bad.sum(), 1)
        log(f"  epoch {ep + 1}: loss {tot / len(Xt):.4f}  val acc {acc:.3f}  wrong-junction recall {rec_bad:.3f}")
    return model


def train_gnn(train, val, cnn, epochs=40, lr=1e-3, hidden=128, layers=4, log=print):
    # attach CNN scores to the junction features (re-run feature build with the trained CNN)
    def rebuild(ex):
        sim = ex["sim"]
        _, g, *_ = prepare(sim.lengths, sim.contacts, sim.draft, cnn=cnn, bin_size=ex["bin_size"],
                           true_adj_fn=lambda cs: adjacencies_from_truth(cs, sim.truth))
        return g

    tg = [rebuild(e) for e in train]
    vg = [rebuild(e) for e in val]
    ys = np.concatenate([g.edge_y for g in tg])
    counts = np.bincount(ys, minlength=5).astype(float)
    w = torch.tensor((counts.sum() / (5 * np.maximum(counts, 1))) ** 0.5, dtype=torch.float32)
    log(f"GNN: {len(ys)} train edges, class counts {counts.astype(int).tolist()}")
    model = EdgeGNN(N_NODE_FEATS, N_EDGE_FEATS, hidden, layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    T = lambda g: (torch.from_numpy(g.node_x), torch.from_numpy(g.edge_index), torch.from_numpy(g.edge_x),
                   torch.from_numpy(g.edge_y))
    tg_t, vg_t = [T(g) for g in tg], [T(g) for g in vg]
    for ep in range(epochs):
        model.train()
        tot = 0
        for j in np.random.permutation(len(tg_t)):
            nx, ei, ex, y = tg_t[j]
            loss = F.cross_entropy(model(nx, ei, ex), y, weight=w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            model.eval()
            tp = fp = fn = 0
            with torch.no_grad():
                for nx, ei, ex, y in vg_t:
                    pr = model(nx, ei, ex).argmax(-1)
                    tp += ((pr == y) & (y > 0)).sum().item()
                    fp += ((pr != y) & (pr > 0)).sum().item()
                    fn += ((pr != y) & (y > 0)).sum().item()
            log(f"  epoch {ep + 1}: loss {tot / len(tg_t):.4f}  val link precision {tp / max(tp + fp, 1):.3f} "
                f"recall {tp / max(tp + fn, 1):.3f}")
    return model


def main(args, log=print):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    t0 = time.time()
    n_val = max(2, args.n_genomes // 10)
    seeds = list(range(args.seed * 100_000, args.seed * 100_000 + args.n_genomes))
    log(f"Simulating {args.n_genomes} genomes ...")
    data = make_examples(seeds, log=log)
    train, val = data[n_val:], data[:n_val]
    cnn = train_cnn(train, val, epochs=args.cnn_epochs, log=log)
    gnn = train_gnn(train, val, cnn, epochs=args.gnn_epochs, hidden=args.hidden, layers=args.layers, log=log)
    save(args.out, cnn, gnn, {"hidden": args.hidden, "layers": args.layers, "cnn_ch": 32})
    log(f"saved {args.out}  ({time.time() - t0:.0f}s)")
