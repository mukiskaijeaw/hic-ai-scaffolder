#!/usr/bin/env python3
"""Convert read-name-grouped Hi-C alignments (bwa mem -5SP) into
   (1) a 4DN .pairs file for hicscaf and
   (2) a Juicer merged_nodups.txt for 3D-DNA.

usage: samtools view -F 0x904 hic.bam | python scripts/bam2pairs.py --mapq 10 -p out.pairs.gz -m merged_nodups.txt
"""
import argparse
import gzip
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--mapq", type=int, default=10)
ap.add_argument("-p", "--pairs", required=True)
ap.add_argument("-m", "--mnd", help="merged_nodups.txt for 3D-DNA")
a = ap.parse_args()

op = gzip.open if a.pairs.endswith(".gz") else open
fp = op(a.pairs, "wt")
fp.write("## pairs format v1.0\n#columns: readID chr1 pos1 chr2 pos2 strand1 strand2\n")
fm = open(a.mnd, "w") if a.mnd else None


def five_prime(flag, pos, cigar):
    """1-based 5' position of the read (reverse reads: end of alignment)."""
    if not flag & 16:
        return pos
    import re
    ref_len = sum(int(n) for n, op in re.findall(r"(\d+)([MDN=X])", cigar))
    return pos + ref_len - 1


prev = None
n_in = n_out = 0
for line in sys.stdin:
    if line.startswith("@"):
        continue
    f = line.split("\t", 11)
    n_in += 1
    rec = (f[0], int(f[1]), f[2], int(f[3]), int(f[4]), f[5])
    if prev is None or prev[0] != rec[0]:
        prev = rec
        continue
    r1, r2, prev = prev, rec, None
    if r1[2] == "*" or r2[2] == "*" or min(r1[4], r2[4]) < a.mapq:
        continue
    p1 = five_prime(r1[1], r1[3], r1[5]); p2 = five_prime(r2[1], r2[3], r2[5])
    s1 = "-" if r1[1] & 16 else "+"; s2 = "-" if r2[1] & 16 else "+"
    fp.write(f"{r1[0]}\t{r1[2]}\t{p1}\t{r2[2]}\t{p2}\t{s1}\t{s2}\n")
    if fm:
        # Juicer long format: str1 chr1 pos1 frag1 str2 chr2 pos2 frag2 mapq1 cigar1 seq1 mapq2 cigar2 seq2 name1 name2
        # 3D-DNA only uses chr/pos/mapq; chr1 must sort before chr2.
        x, y = (r1, p1, s1), (r2, p2, s2)
        if (x[0][2], x[1]) > (y[0][2], y[1]):
            x, y = y, x
        st = lambda s: 0 if s == "+" else 16
        fm.write(f"{st(x[2])} {x[0][2]} {x[1]} 0 {st(y[2])} {y[0][2]} {y[1]} 1 {x[0][4]} - - {y[0][4]} - - {r1[0]} {r1[0]}\n")
    n_out += 1
fp.close()
if fm:
    fm.close()
print(f"bam2pairs: {n_in:,} alignments -> {n_out:,} pairs (mapq>={a.mapq})", file=sys.stderr)
