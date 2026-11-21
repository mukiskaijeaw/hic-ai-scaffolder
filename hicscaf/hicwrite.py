"""Write a new .hic for a scaffolding, so it can be opened in Juicebox together with its .assembly.

The file uses Juicebox Assembly Tools (JBAT) "assembly mode": one pseudo-chromosome called
`assembly` that is all scaffolds laid end to end, in the order and orientation of the .assembly
written next to it. In Juicebox: File > Open > this .hic, then Assembly > Import Map Assembly >
the .assembly. Needs hictkpy + pandas + pyarrow, all installed with hicscaf.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

RESOLUTIONS = (2_500_000, 1_000_000, 500_000, 250_000, 100_000, 50_000, 25_000, 10_000, 5_000, 2_000, 1_000)


def can_write_hic() -> bool:
    try:
        import hictkpy  # noqa: F401
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def layout_positions(cs, scafs, mapped):
    """Contact positions in the end-to-end layout of `scafs` (bp)."""
    N = len(cs.comps)
    off = np.zeros(N, np.int64)
    strand = np.ones(N, np.int64)
    pos = 0
    for s in scafs:
        for c, st in s.parts:
            i = cs.index[c]
            off[i], strand[i] = pos, st
            pos += c.length
    L = cs.lengths
    a, pa, b, pb, w = mapped
    x = np.where(strand[a] > 0, off[a] + pa, off[a] + L[a] - 1 - pa)
    y = np.where(strand[b] > 0, off[b] + pb, off[b] + L[b] - 1 - pb)
    return x, y, w, int(pos)


def write_hic(path, cs, mapped, scafs, min_resolution: int = 5_000, genome: str = "hicscaf") -> list[int]:
    """Write `path` (.hic) in assembly mode. Returns the resolutions written."""
    import hictkpy
    import pandas as pd

    x, y, w, total = layout_positions(cs, scafs, mapped)
    res = sorted(r for r in RESOLUTIONS if r >= min_resolution and r < total) or [min_resolution]
    base = res[0]
    b1, b2 = x // base, y // base
    lo, hi = np.minimum(b1, b2), np.maximum(b1, b2)
    nbin = (total + base - 1) // base
    key = lo * nbin + hi
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv, weights=w)
    s1, s2 = (uk // nbin).astype(np.int64) * base, (uk % nbin).astype(np.int64) * base
    df = pd.DataFrame({"chrom1": "assembly", "start1": s1, "end1": np.minimum(s1 + base, total),
                       "chrom2": "assembly", "start2": s2, "end2": np.minimum(s2 + base, total),
                       "count": np.rint(cnt).astype(np.int32)})
    df = df[df["count"] > 0]
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.hic")
    if tmp.exists():
        tmp.unlink()
    with tempfile.TemporaryDirectory() as td:
        wr = hictkpy.hic.FileWriter(str(tmp), {"assembly": int(total)}, res, assembly=genome, tmpdir=td)
        wr.add_pixels(df)
        wr.finalize()
    os.replace(tmp, path)
    return res
