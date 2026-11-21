"""Synthetic training data with free labels.

Take a genome (real reference lengths, or random), cut chromosomes into contigs,
shuffle and flip them, simulate Hi-C contacts with a power-law distance decay,
and build a *draft* scaffolding that contains YaHS/3D-DNA-like errors
(inversions, relocations, chromosome fusions, missed joins). The model learns
to turn the draft back into the truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Component, Contacts, Scaffold


@dataclass
class SimData:
    lengths: dict[str, int]                       # contig -> length
    contacts: Contacts                            # contig coordinates
    truth: dict[str, tuple[str, int, int, int]]   # contig -> (chrom, start, end, strand)
    draft: list[Scaffold]                         # corrupted scaffolding
    true_scaffolds: list[Scaffold]


# ----------------------------------------------------------------- genome
def random_genome(rng, n_chrom=(3, 8), chrom_len=(1_500_000, 6_000_000)) -> dict[str, int]:
    k = int(rng.integers(*n_chrom, endpoint=True))
    return {f"chr{i + 1}": int(rng.integers(*chrom_len)) for i in range(k)}


def cut_contigs(rng, chroms: dict[str, int], mean_len=250_000, sigma=0.8, min_len=15_000):
    """Return list of (chrom, start, end, strand) in genome order."""
    out = []
    mu = np.log(mean_len) - sigma**2 / 2
    for ch, L in chroms.items():
        pos = 0
        while pos < L:
            l = max(min_len, int(rng.lognormal(mu, sigma)))
            end = min(L, pos + l)
            if L - end < min_len:
                end = L
            out.append((ch, pos, end, 1 if rng.random() < 0.5 else -1))
            pos = end
    return out


# ---------------------------------------------------------------- contacts
def simulate_genome_contacts(rng, chroms: dict[str, int], n_pairs: int, alpha=1.05, trans_frac=0.03,
                             min_dist=1_000, bias_bin=10_000, bias_sigma=0.4, compartment_strength=0.0,
                             repeat_frac=0.0):
    """Sample read-pair positions in genome coordinates.

    cis pairs: distance d ~ P(d) ∝ d^-alpha on [min_dist, chrom_len]; trans pairs uniform.
    A log-normal per-bin coverage bias mimics restriction-site / mappability bias.
    compartment_strength in [0,1): long-range / trans contacts between A and B compartments
    are depleted (checkerboard pattern that can mislead order inference).
    repeat_frac: fraction of the genome made unmappable (contacts there are dropped).
    """
    if compartment_strength > 0 or repeat_frac > 0:
        n_raw = int(n_pairs / max(1e-3, (1 - repeat_frac) ** 2 * (1 - compartment_strength / 2)))
        names, ch1, pos1, ch2, pos2 = simulate_genome_contacts(
            rng, chroms, n_raw, alpha, trans_frac, min_dist, bias_bin, bias_sigma)
        keep = np.ones(len(ch1), bool)
        seg = 50_000
        L = np.array([chroms[n] for n in names])
        nseg = (L + seg - 1) // seg
        offs = np.concatenate([[0], np.cumsum(nseg)])
        g1, g2 = offs[ch1] + pos1 // seg, offs[ch2] + pos2 // seg
        if compartment_strength > 0:
            # compartments: runs of 50 kb segments, run length ~ geometric (mean 1 Mb)
            comp = np.cumsum(rng.random(offs[-1]) < 1 / 20) % 2
            far = (ch1 != ch2) | (np.abs(pos1 - pos2) > 300_000)
            keep &= ~(far & (comp[g1] != comp[g2]) & (rng.random(len(ch1)) < compartment_strength))
        if repeat_frac > 0:
            rep = np.zeros(offs[-1] * 5, bool)  # 10 kb resolution
            rep[rng.random(len(rep)) < repeat_frac] = True
            keep &= ~rep[offs[ch1] * 5 + pos1 // 10_000] & ~rep[offs[ch2] * 5 + pos2 // 10_000]
        m = np.nonzero(keep)[0][:n_pairs]
        return names, ch1[m], pos1[m], ch2[m], pos2[m]
    names = list(chroms)
    L = np.array([chroms[n] for n in names], dtype=np.int64)
    # per-bin bias
    nb = (L + bias_bin - 1) // bias_bin
    offs = np.concatenate([[0], np.cumsum(nb)])
    w = rng.lognormal(0, bias_sigma, offs[-1])
    w /= w.sum()
    b1 = rng.choice(offs[-1], size=n_pairs, p=w)
    ch1 = np.searchsorted(offs, b1, side="right") - 1
    pos1 = (b1 - offs[ch1]) * bias_bin + rng.integers(0, bias_bin, n_pairs)
    pos1 = np.minimum(pos1, L[ch1] - 1)

    ch2 = ch1.copy()
    pos2 = np.empty_like(pos1)
    trans = rng.random(n_pairs) < trans_frac
    # cis: inverse-CDF sampling of a truncated power law
    cis = ~trans
    n = cis.sum()
    u = rng.random(n)
    lo, hi = float(min_dist), L[ch1[cis]].astype(float)
    if abs(alpha - 1) < 1e-6:
        d = lo * (hi / lo) ** u
    else:
        a = 1 - alpha
        d = (lo**a + u * (hi**a - lo**a)) ** (1 / a)
    sign = np.where(rng.random(n) < 0.5, -1, 1)
    p2 = pos1[cis] + sign * d.astype(np.int64)
    # reflect into chromosome
    Lc = L[ch1[cis]]
    p2 = np.where(p2 < 0, -p2, p2)
    p2 = np.where(p2 >= Lc, 2 * (Lc - 1) - p2, p2)
    pos2[cis] = np.clip(p2, 0, Lc - 1)
    # trans
    nt = trans.sum()
    if nt:
        ch2[trans] = rng.choice(len(names), size=nt, p=L / L.sum())
        pos2[trans] = (rng.random(nt) * L[ch2[trans]]).astype(np.int64)
    return names, ch1, pos1, ch2, pos2


def genome_to_contig_coords(contigs, chrom_names, ch, pos):
    """contigs: list of (chrom, start, end, strand); returns contig index and local position."""
    cidx = np.empty(len(ch), np.int32)
    local = np.empty(len(ch), np.int64)
    by_chrom: dict[str, list[int]] = {}
    for i, c in enumerate(contigs):
        by_chrom.setdefault(c[0], []).append(i)
    for k, cname in enumerate(chrom_names):
        m = ch == k
        if not m.any():
            continue
        ids = np.array(by_chrom[cname])
        starts = np.array([contigs[i][1] for i in ids])
        j = np.searchsorted(starts, pos[m], side="right") - 1
        idx = ids[j]
        st = np.array([contigs[i][1] for i in ids])[j]
        en = np.array([contigs[i][2] for i in ids])[j]
        strand = np.array([contigs[i][3] for i in ids])[j]
        cidx[m] = idx
        local[m] = np.where(strand > 0, pos[m] - st, en - 1 - pos[m])
    return cidx, local


# ------------------------------------------------------------- draft errors
def corrupt(rng, true_scafs: list[Scaffold], p_flip=0.04, p_relocate=0.03, p_swap=0.03, p_fuse=0.3,
            p_break=0.02, p_unplaced_small=0.5, small_len=40_000) -> list[Scaffold]:
    """Inject the kinds of errors Hi-C scaffolders make."""
    scafs = [list(s.parts) for s in true_scafs]
    unplaced = []
    # small contigs often left unplaced
    for s in scafs:
        keep = []
        for c, st in s:
            if c.length < small_len and rng.random() < p_unplaced_small:
                unplaced.append([(c, 1 if rng.random() < 0.5 else -1)])
            else:
                keep.append((c, st))
        s[:] = keep
    scafs = [s for s in scafs if s]
    # orientation errors
    for s in scafs:
        for i in range(len(s)):
            if rng.random() < p_flip:
                s[i] = (s[i][0], -s[i][1])
    # local swaps
    for s in scafs:
        for i in range(len(s) - 1):
            if rng.random() < p_swap:
                s[i], s[i + 1] = s[i + 1], s[i]
    # relocations (possibly to another scaffold)
    moved = []
    for s in scafs:
        i = 0
        while i < len(s):
            if len(s) > 1 and rng.random() < p_relocate:
                moved.append(s.pop(i))
            else:
                i += 1
    scafs = [s for s in scafs if s]
    for item in moved:
        t = scafs[int(rng.integers(len(scafs)))]
        t.insert(int(rng.integers(len(t) + 1)), item)
    # missed joins (scaffold broken)
    out = []
    for s in scafs:
        cur = [s[0]]
        for item in s[1:]:
            if rng.random() < p_break:
                out.append(cur)
                cur = []
            cur.append(item)
        out.append(cur)
    scafs = out
    # chromosome fusions (misjoins between chromosomes)
    rng.shuffle(scafs)
    fused = []
    i = 0
    while i < len(scafs):
        if i + 1 < len(scafs) and rng.random() < p_fuse / max(1, len(scafs) / 4):
            b = scafs[i + 1]
            if rng.random() < 0.5:
                b = [(c, -st) for c, st in reversed(b)]
            fused.append(scafs[i] + b)
            i += 2
        else:
            fused.append(scafs[i])
            i += 1
    scafs = fused + unplaced
    # random global orientation of each scaffold
    res = []
    for k, s in enumerate(scafs):
        if rng.random() < 0.5:
            s = [(c, -st) for c, st in reversed(s)]
        res.append(Scaffold(f"scaffold_{k + 1}", s))
    return res


# ----------------------------------------------------------------- driver
def simulate(seed: int = 0, chroms: dict[str, int] | None = None, pairs_per_mb: int = 40_000,
             mean_contig: int | None = None, error_scale: float = 1.0, realism: float = 1.0) -> SimData:
    """realism scales the nasty parts: compartments, repeats, short contigs (0 = clean toy data)."""
    rng = np.random.default_rng(seed)
    chroms = chroms or random_genome(rng)
    mean_contig = mean_contig or int(rng.choice([80_000, 150_000, 300_000]))
    ctgs = cut_contigs(rng, chroms, mean_len=mean_contig, min_len=int(15_000 - 10_000 * min(realism, 1)))
    perm = rng.permutation(len(ctgs))  # contig names shuffled w.r.t. genome order
    cnames = [""] * len(ctgs)
    for new, old in enumerate(perm):
        cnames[old] = f"ctg{new + 1:05d}"
    lengths = {cnames[i]: c[2] - c[1] for i, c in enumerate(ctgs)}
    lengths = dict(sorted(lengths.items()))
    truth = {cnames[i]: c for i, c in enumerate(ctgs)}

    G = sum(chroms.values())
    n_pairs = int(pairs_per_mb * G / 1e6 * rng.uniform(0.5, 1.5))
    chrom_names, ch1, pos1, ch2, pos2 = simulate_genome_contacts(
        rng, chroms, n_pairs, alpha=rng.uniform(0.8, 1.4), trans_frac=rng.uniform(0.01, 0.1),
        bias_sigma=0.3 + 0.4 * realism, compartment_strength=min(0.9, realism * rng.uniform(0.2, 0.7)),
        repeat_frac=min(0.5, realism * rng.uniform(0.02, 0.15)))
    i1, l1 = genome_to_contig_coords(ctgs, chrom_names, ch1, pos1)
    i2, l2 = genome_to_contig_coords(ctgs, chrom_names, ch2, pos2)
    names = list(lengths)
    remap = np.array([names.index(cnames[i]) for i in range(len(ctgs))], np.int32)
    contacts = Contacts(names, remap[i1], l1, remap[i2], l2)

    true_scafs = []
    for ch in chroms:
        parts = [(Component(cnames[i], 0, c[2] - c[1]), c[3]) for i, c in enumerate(ctgs) if c[0] == ch]
        true_scafs.append(Scaffold(ch, parts))
    e = error_scale * rng.uniform(0.5, 1.5)
    draft = corrupt(rng, true_scafs, p_flip=0.04 * e, p_relocate=0.03 * e, p_swap=0.03 * e,
                    p_fuse=0.3 * e, p_break=0.02 * e)
    return SimData(lengths, contacts, truth, draft, true_scafs)


# ------------------------------------------------------ reads from reference
def simulate_hic_reads(ref: dict[str, str], n_pairs: int, out_prefix: str, read_len=150, err=0.005, seed=0,
                       alpha=1.05, trans_frac=0.03):
    """Write paired FASTQ of synthetic Hi-C reads from a real reference.

    Lets you benchmark YaHS / 3D-DNA / hicscaf end-to-end without downloading Hi-C data.
    Each read pair = read_len bp starting at each end of a simulated ligation junction.
    """
    import gzip

    rng = np.random.default_rng(seed)
    chroms = {k: len(v) for k, v in ref.items() if len(v) > 4 * read_len}
    names, ch1, pos1, ch2, pos2 = simulate_genome_contacts(rng, chroms, n_pairs, alpha=alpha, trans_frac=trans_frac)
    comp = str.maketrans("ACGTN", "TGCAN")
    bases = np.array(list("ACGT"))
    qual = "I" * read_len
    with gzip.open(f"{out_prefix}_R1.fastq.gz", "wt") as f1, gzip.open(f"{out_prefix}_R2.fastq.gz", "wt") as f2:
        for i in range(n_pairs):
            reads = []
            for ch, p in ((ch1[i], pos1[i]), (ch2[i], pos2[i])):
                s = ref[names[ch]]
                p = int(min(max(0, p), len(s) - read_len))
                r = s[p : p + read_len].upper()
                if rng.random() < 0.5:
                    r = r.translate(comp)[::-1]
                if err > 0:
                    r = list(r)
                    for j in np.nonzero(rng.random(read_len) < err)[0]:
                        r[j] = bases[rng.integers(4)]
                    r = "".join(r)
                reads.append(r)
            f1.write(f"@hic{i}/1\n{reads[0]}\n+\n{qual}\n")
            f2.write(f"@hic{i}/2\n{reads[1]}\n+\n{qual}\n")


def cut_reference_into_contigs(ref: dict[str, str], seed=0, mean_len=250_000, min_len=15_000):
    """Fragment a real reference into shuffled/flipped contigs (truth is known exactly)."""
    from .io import revcomp

    rng = np.random.default_rng(seed)
    ctgs = cut_contigs(rng, {k: len(v) for k, v in ref.items()}, mean_len=mean_len, min_len=min_len)
    perm = rng.permutation(len(ctgs))
    seqs, truth = {}, {}
    for new, old in enumerate(perm):
        ch, s, e, st = ctgs[old]
        name = f"ctg{new + 1:05d}"
        seq = ref[ch][s:e]
        seqs[name] = seq if st > 0 else revcomp(seq)
        truth[name] = (ch, s, e, st)
    return dict(sorted(seqs.items())), truth
