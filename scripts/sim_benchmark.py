#!/usr/bin/env python3
"""Held-out synthetic benchmark: draft (YaHS-like) vs hicscaf hybrid vs hicscaf from scratch.

usage: python scripts/sim_benchmark.py --model models/hicscaf.pt --n 10
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hicscaf.evaluate import COLUMNS, evaluate, format_table
from hicscaf.models import load
from hicscaf.scaffold import correct, heuristic_scaffold
from hicscaf.simulate import simulate

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/hicscaf.pt")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed0", type=int, default=9_000_000)
ap.add_argument("--pairs-per-mb", type=int, default=30_000)
ap.add_argument("--min-prob", type=float, default=0.5)
a = ap.parse_args()

cnn, gnn, _ = load(a.model)
acc = defaultdict(lambda: defaultdict(list))
for s in range(a.seed0, a.seed0 + a.n):
    sim = simulate(seed=s, pairs_per_mb=a.pairs_per_mb)
    runs = {
        "draft (YaHS-like errors)": sim.draft,
        "greedy end-contact baseline (no AI)": heuristic_scaffold(sim.lengths, sim.contacts),
        "hicscaf hybrid (draft+AI)": correct(sim.lengths, sim.contacts, sim.draft, cnn, gnn, a.min_prob),
        "hicscaf from scratch": correct(sim.lengths, sim.contacts, None, cnn, gnn, a.min_prob),
    }
    for k, v in runs.items():
        for m, x in evaluate(v, sim.truth).items():
            acc[k][m].append(x)
rows = {k: {m: (float(np.mean(v)) if m in ("precision", "recall", "F1") else int(round(np.mean(v))))
            for m, v in d.items()} for k, d in acc.items()}
print(f"mean over {a.n} held-out synthetic genomes ({a.pairs_per_mb:,} pairs/Mb)\n")
print(format_table(rows))
