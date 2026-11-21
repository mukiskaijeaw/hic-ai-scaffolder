"""Use outputs that YaHS / 3D-DNA (or DNA Zoo, DToL, VGP/GenomeArk) already produced - no raw reads needed.

Supported inputs
  draft scaffolding : .assembly (3D-DNA / Juicebox, incl. DNA Zoo *.rawchrom.assembly) or .agp (YaHS, DToL, VGP)
  ground truth      : a curated .review.assembly / curated AGP / or a reference (via minimap2 PAF)
  contacts          : .hic  (Juicer; via the `hicstraw` package or juicer_tools.jar `dump`)
                      .mcool/.cool (via the `cooler` package)
                      text dump `pos1 pos2 count` or `chr1 pos1 chr2 pos2 count`

Coordinates of a contact matrix are in *scaffold* space. A Layout describes where each contig piece
lies in that space so binned contacts can be mapped back to contig coordinates:
  * "assembly" layout - Juicebox Assembly Tools .hic: one pseudo-chromosome "assembly" that is all
    scaffolds of the .assembly concatenated without gaps (coordinates divided by `scale` for genomes > 2.1 Gb)
  * "agp" layout      - .hic/.mcool built on the scaffold FASTA: one chromosome per AGP object,
    coordinates exactly as in the AGP (gaps included)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .io import Component, Contacts, Scaffold


# ------------------------------------------------------------------ .assembly
def read_assembly(path) -> tuple[list[Scaffold], dict[str, int]]:
    """Parse a 3D-DNA / Juicebox .assembly. Fragments `ctg:::fragment_2`, `ctg:::debris` are mapped
    back to coordinates on the original contig `ctg` (fragments are listed in order along each contig)."""
    frags, offset, rows = {}, defaultdict(int), []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                name, idx, length = line[1:].split()
                base = name.split(":::")[0]
                b = offset[base]
                frags[int(idx)] = Component(base, b, b + int(length))
                offset[base] += int(length)
            elif line.strip():
                rows.append([int(x) for x in line.split()])
    scafs = [Scaffold(f"HiC_scaffold_{k}", [(frags[abs(i)], 1 if i > 0 else -1) for i in row])
             for k, row in enumerate(rows, 1)]
    return scafs, dict(offset)


# --------------------------------------------------------------------- layout
@dataclass
class Layout:
    """chrom -> arrays (obj_beg, obj_end, contig_idx, contig_beg, strand) sorted by obj_beg."""
    names: list[str]                         # contig names
    blocks: dict[str, tuple[np.ndarray, ...]]

    def to_contig(self, chrom: str, pos: np.ndarray):
        if chrom not in self.blocks:
            return np.full(len(pos), -1, np.int32), np.zeros(len(pos), np.int64)
        ob, oe, ci, cb, st = self.blocks[chrom]
        j = np.clip(np.searchsorted(ob, pos, side="right") - 1, 0, len(ob) - 1)
        inside = (pos >= ob[j]) & (pos < oe[j])
        local = np.where(st[j] > 0, cb[j] + (pos - ob[j]), cb[j] + (oe[j] - 1 - pos))
        return np.where(inside, ci[j], -1).astype(np.int32), local.astype(np.int64)


def _layout(rows_by_chrom, names):
    idx = {n: i for i, n in enumerate(names)}
    blocks = {}
    for ch, rows in rows_by_chrom.items():
        rows.sort()
        a = np.array([(ob, oe, idx[c.contig], c.beg, st) for ob, oe, c, st in rows], np.int64).reshape(-1, 5)
        blocks[ch] = tuple(a[:, k] for k in range(5))
    return Layout(names, blocks)


def layout_from_assembly(scafs: list[Scaffold], names: list[str], chrom="assembly") -> Layout:
    rows, pos = defaultdict(list), 0
    for s in scafs:
        for c, st in s.parts:
            rows[chrom].append((pos, pos + c.length, c, st))
            pos += c.length
    return _layout(rows, names)


def layout_from_agp(path, names: list[str]) -> Layout:
    rows = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if f[4] in ("N", "U"):
                continue
            c = Component(f[5], int(f[6]) - 1, int(f[7]))
            rows[f[0]].append((int(f[1]) - 1, int(f[2]), c, -1 if f[8] == "-" else 1))
    return _layout(rows, names)


# -------------------------------------------------------------- matrix readers
def _records_hicstraw(path, resolution, chroms=None):
    import hicstraw  # pip install hic-straw

    hic = hicstraw.HiCFile(str(path))
    cs = [c for c in hic.getChromosomes() if c.name.lower() != "all" and (chroms is None or c.name in chroms)]
    for i, c1 in enumerate(cs):
        for c2 in cs[i:]:
            mzd = hic.getMatrixZoomData(c1.name, c2.name, "observed", "NONE", "BP", resolution)
            r = mzd.getRecords(0, c1.length, 0, c2.length)
            if r:
                yield c1.name, c2.name, np.array([[x.binX, x.binY, x.counts] for x in r], np.float64)


def _records_juicer_tools(path, resolution, chroms, jar):
    tmp = tempfile.mkdtemp()
    try:
        for i, c1 in enumerate(chroms):
            for c2 in chroms[i:]:
                out = os.path.join(tmp, "d.txt")
                subprocess.run(["java", f"-Xmx{os.environ.get('JAVA_MEM', '8g')}", "-jar", jar, "dump", "observed",
                                "NONE", str(path), c1, c2, "BP", str(resolution), out],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.getsize(out):
                    yield c1, c2, np.loadtxt(out, ndmin=2)
    finally:
        shutil.rmtree(tmp)


def _records_cooler(path, resolution):
    import cooler  # pip install cooler

    uri = f"{path}::/resolutions/{resolution}" if str(path).endswith(".mcool") else str(path)
    c = cooler.Cooler(uri)
    bins = c.bins()[:][["chrom", "start"]]
    pix = c.pixels()[:]
    ch1 = bins["chrom"].values[pix["bin1_id"].values]
    ch2 = bins["chrom"].values[pix["bin2_id"].values]
    s1 = bins["start"].values[pix["bin1_id"].values]
    s2 = bins["start"].values[pix["bin2_id"].values]
    cnt = pix["count"].values
    for a in np.unique(ch1):
        for b in np.unique(ch2[ch1 == a]):
            m = (ch1 == a) & (ch2 == b)
            yield str(a), str(b), np.stack([s1[m], s2[m], cnt[m]], 1).astype(np.float64)


def _records_text(path):
    rows = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) == 3:
                rows[("assembly", "assembly")].append((float(f[0]), float(f[1]), float(f[2])))
            elif len(f) >= 5:
                rows[(f[0], f[2])].append((float(f[1]), float(f[3]), float(f[4])))
    for (a, b), r in rows.items():
        yield a, b, np.array(r)


def load_matrix_contacts(path, layout: Layout, resolution: int = 5000, juicer_tools: str | None = None,
                         scale: int = 1) -> Contacts:
    """Read a binned contact matrix and express it as weighted contacts in contig coordinates."""
    p = str(path)
    if p.endswith(".hic"):
        try:
            import hicstraw  # noqa: F401
            recs = _records_hicstraw(p, resolution, set(layout.blocks))
        except ImportError:
            jar = juicer_tools or os.environ.get("JUICER_TOOLS")
            if not jar:
                raise SystemExit("reading .hic needs `pip install hic-straw` or --juicer-tools /path/juicer_tools.jar")
            recs = _records_juicer_tools(p, resolution, sorted(layout.blocks), jar)
    elif p.endswith((".mcool", ".cool")):
        recs = _records_cooler(p, resolution)
    else:
        recs = _records_text(p)
    C1, P1, C2, P2, W = [], [], [], [], []
    for ch1, ch2, r in recs:
        if len(r) == 0:
            continue
        x = (r[:, 0] * scale).astype(np.int64) + resolution // 2
        y = (r[:, 1] * scale).astype(np.int64) + resolution // 2
        a, pa = layout.to_contig(ch1, x)
        b, pb = layout.to_contig(ch2, y)
        m = (a >= 0) & (b >= 0) & (r[:, 2] > 0)
        C1.append(a[m]); P1.append(pa[m]); C2.append(b[m]); P2.append(pb[m]); W.append(r[m, 2])
    cat = lambda xs, t: np.concatenate(xs).astype(t) if xs else np.zeros(0, t)
    return Contacts(layout.names, cat(C1, np.int32), cat(P1, np.int64), cat(C2, np.int32), cat(P2, np.int64),
                    cat(W, np.float32), resolution * scale)


def load_scaffolds(path) -> tuple[list[Scaffold], dict[str, int]]:
    """.assembly or .agp -> scaffolds + contig lengths (as far as the file tells)."""
    if str(path).endswith(".assembly"):
        return read_assembly(path)
    from .io import read_agp
    scafs = read_agp(path)
    lengths: dict[str, int] = {}
    for s in scafs:
        for c, _ in s.parts:
            lengths[c.contig] = max(lengths.get(c.contig, 0), c.end)
    return scafs, lengths


def write_assembly(scafs: list[Scaffold], lengths: dict[str, int], path) -> None:
    """Write a Juicebox .assembly. Pieces of a contig become `ctg:::fragment_k` (gaps between pieces
    become `:::debris`) so the file loads in Juicebox Assembly Tools on top of the original contigs."""
    pieces: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for s in scafs:
        for c, _ in s.parts:
            pieces[c.contig].append((c.beg, c.end))
    frag_id: dict[tuple[str, int, int], int] = {}
    lines, k = [], 0
    for name in list(lengths) + [n for n in pieces if n not in lengths]:
        iv = sorted(pieces.get(name, []))
        L = lengths.get(name, iv[-1][1] if iv else 0)
        whole = len(iv) == 1 and iv[0] == (0, L)
        cuts, pos = [], 0
        for b, e in iv:
            if b > pos:
                cuts.append((pos, b, True))
            cuts.append((b, e, False)); pos = e
        if pos < L:
            cuts.append((pos, L, True))
        for j, (b, e, debris) in enumerate(cuts, 1):
            k += 1
            label = name if whole else f"{name}:::fragment_{j}" + (":::debris" if debris else "")
            lines.append(f">{label} {k} {e - b}")
            if not debris:
                frag_id[(name, b, e)] = k
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
        for s in scafs:
            fh.write(" ".join(str(frag_id[(c.contig, c.beg, c.end)] * st) for c, st in s.parts) + "\n")
