#!/usr/bin/env python3
"""Plot a genome-wide Hi-C contact map for a scaffolding (AGP) - needs matplotlib.

usage: python scripts/plot_contact_map.py --contigs contigs.fa --pairs hic.pairs.gz --agp yahs.agp -o yahs.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from hicscaf.features import ComponentSet, map_contacts
from hicscaf.io import Contacts, add_missing_components, read_agp, read_lengths

ap = argparse.ArgumentParser()
ap.add_argument("--contigs", required=True); ap.add_argument("--pairs", required=True)
ap.add_argument("--agp", required=True); ap.add_argument("--bins", type=int, default=1000)
ap.add_argument("--title"); ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lengths = read_lengths(a.contigs)
scafs = add_missing_components(read_agp(a.agp), lengths)
scafs.sort(key=lambda s: -sum(c.length for c, _ in s.parts))
cs = ComponentSet.from_scaffolds(scafs)
ca, pa, cb, pb, w = map_contacts(cs, Contacts.load(a.pairs))
N = len(cs.comps); L = cs.lengths
off = np.zeros(N, np.int64); strand = np.ones(N, np.int64); bounds = []
pos = 0
for s in scafs:
    for c, st in s.parts:
        i = cs.index[c]; off[i] = pos; strand[i] = st; pos += c.length
    bounds.append(pos)
x = off[ca] + np.where(strand[ca] > 0, pa, L[ca] - 1 - pa)
y = off[cb] + np.where(strand[cb] > 0, pb, L[cb] - 1 - pb)
b = max(1, pos // a.bins)
H = np.zeros((pos // b + 1,) * 2)
np.add.at(H, (x // b, y // b), w); np.add.at(H, (y // b, x // b), w)
plt.figure(figsize=(8, 8))
plt.imshow(np.log1p(H), cmap="Reds", interpolation="none")
for bd in bounds[:-1]:
    if bd / pos > 0.005:
        plt.axhline(bd / b, color="k", lw=0.3); plt.axvline(bd / b, color="k", lw=0.3)
plt.title(a.title or Path(a.agp).name); plt.xticks([]); plt.yticks([])
plt.tight_layout(); plt.savefig(a.out, dpi=150)
print("wrote", a.out)
