#!/usr/bin/env python3
"""Convert a 3D-DNA / Juicebox .assembly file into AGP (in original-contig coordinates).

3D-DNA may split contigs into 'name:::fragment_N' (and ':::debris') pieces; fragments are laid out
consecutively along the original contig, so their coordinates are recovered from cumulative lengths.

usage: python scripts/assembly2agp.py contigs.FINAL.assembly > 3ddna.agp
"""
import sys
from collections import defaultdict

frags = {}             # index -> (contig, beg, end)
offset = defaultdict(int)
scaffolds = []
with open(sys.argv[1]) as fh:
    for line in fh:
        if line.startswith(">"):
            name, idx, length = line[1:].split()
            base = name.split(":::")[0]
            beg = offset[base]
            frags[int(idx)] = (base, beg, beg + int(length))
            offset[base] += int(length)
        elif line.strip():
            scaffolds.append([int(x) for x in line.split()])

gap = 500
print("##agp-version 2.1")
for k, scaf in enumerate(scaffolds, 1):
    pos, part = 0, 1
    for i, sidx in enumerate(scaf):
        c, b, e = frags[abs(sidx)]
        if i:
            print(f"HiC_scaffold_{k}\t{pos + 1}\t{pos + gap}\t{part}\tU\t{gap}\tscaffold\tyes\tproximity_ligation")
            pos += gap; part += 1
        print(f"HiC_scaffold_{k}\t{pos + 1}\t{pos + e - b}\t{part}\tW\t{c}\t{b + 1}\t{e}\t{'-' if sidx < 0 else '+'}")
        pos += e - b; part += 1
