"""Minimal reader for Juicer .hic files (versions 7-9), pure Python + numpy.

Only what hicscaf needs: the header (chromosomes, resolutions) and the raw ("observed", no
normalisation) counts of one chromosome pair at one base-pair resolution. This removes the
dependency on `hic-straw`, which often fails to build on Windows.

Format reference: https://github.com/aidenlab/hic-format/blob/master/HiCFormatV9.md
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np


class _R:
    """Little-endian reader over a bytes buffer."""

    def __init__(self, buf: bytes, pos: int = 0):
        self.b, self.p = buf, pos

    def _u(self, fmt, n):
        v = struct.unpack_from(fmt, self.b, self.p)[0]
        self.p += n
        return v

    def i8(self): return self._u("<b", 1)
    def i16(self): return self._u("<h", 2)
    def i32(self): return self._u("<i", 4)
    def i64(self): return self._u("<q", 8)
    def f32(self): return self._u("<f", 4)
    def f64(self): return self._u("<d", 8)

    def s(self):
        e = self.b.index(b"\0", self.p)
        v = self.b[self.p:e].decode("utf-8", "replace")
        self.p = e + 1
        return v


@dataclass
class Chrom:
    index: int
    name: str
    length: int


@dataclass
class HicFile:
    path: str
    version: int = 0
    genome: str = ""
    chroms: list[Chrom] = field(default_factory=list)
    resolutions: list[int] = field(default_factory=list)
    _index: dict = field(default_factory=dict)  # "i_j" -> (position, size)

    @classmethod
    def open(cls, path):
        h = cls(str(path))
        with open(path, "rb") as fh:
            head = fh.read(1 << 20)
            while True:  # make sure the header (with many chromosomes) is fully in the buffer
                try:
                    h._parse_header(head)
                    break
                except (struct.error, ValueError):
                    more = fh.read(len(head))
                    if not more:
                        raise ValueError(f"{path}: not a valid .hic file")
                    head += more
            fh.seek(h._master)
            foot = fh.read(16)
            r = _R(foot)
            nbytes = r.i64() if h.version >= 9 else r.i32()
            fh.seek(h._master)
            foot = fh.read(nbytes + 16)
        r = _R(foot)
        r.i64() if h.version >= 9 else r.i32()
        for _ in range(r.i32()):
            key = r.s()
            pos = r.i64()
            size = r.i32()
            h._index[key] = (pos, size)
        return h

    def _parse_header(self, buf):
        r = _R(buf)
        magic = r.s()
        if magic != "HIC":
            raise SystemExit(f"{self.path}: not a .hic file")
        self.version = r.i32()
        if self.version < 7:
            raise SystemExit(f"{self.path}: .hic version {self.version} is too old (need 7-9)")
        self._master = r.i64()
        self.genome = r.s()
        if self.version >= 9:
            r.i64(); r.i64()  # normalisation vector index position / length
        for _ in range(r.i32()):
            r.s(); r.s()  # attributes
        self.chroms = []
        for i in range(r.i32()):
            name = r.s()
            length = r.i64() if self.version >= 9 else r.i32()
            self.chroms.append(Chrom(i, name, length))
        self.resolutions = [r.i32() for _ in range(r.i32())]

    # ------------------------------------------------------------------ data
    def records(self, c1: Chrom, c2: Chrom, resolution: int, progress=None) -> np.ndarray:
        """(n, 3) array: binX * resolution, binY * resolution, count (observed, no normalisation)."""
        if c1.index > c2.index:
            c1, c2 = c2, c1
        key = f"{c1.index}_{c2.index}"
        if key not in self._index:
            return np.zeros((0, 3))
        pos, size = self._index[key]
        with open(self.path, "rb") as fh:
            fh.seek(pos)
            r = _R(fh.read(size))
            r.i32(); r.i32()
            blocks = None
            for _ in range(r.i32()):
                unit = r.s()
                r.i32()                     # old index
                r.f32(); r.f32(); r.f32(); r.f32()  # sum, occupied cells, std, 95th percentile
                bin_size = r.i32()
                r.i32(); r.i32()            # block bin count, block column count
                nb = r.i32()
                lst = [(r.i32(), r.i64(), r.i32()) for _ in range(nb)]
                if unit == "BP" and bin_size == resolution:
                    blocks = lst
            if blocks is None:
                raise ValueError(f"resolution {resolution} not in {self.path} (has {self.resolutions})")
            out = []
            for k, (_, bpos, bsize) in enumerate(blocks):
                if progress and k % 20 == 0:
                    progress(k / len(blocks))
                if bsize <= 0:
                    continue
                fh.seek(bpos)
                out.append(self._block(zlib.decompress(fh.read(bsize))))
        if not out:
            return np.zeros((0, 3))
        a = np.concatenate(out)
        a[:, 0] *= resolution
        a[:, 1] *= resolution
        return a

    def _block(self, buf: bytes) -> np.ndarray:
        r = _R(buf)
        n = r.i32()
        if self.version < 7:  # pragma: no cover - rejected in the header already
            a = np.frombuffer(buf, dtype=np.dtype([("x", "<i4"), ("y", "<i4"), ("c", "<f4")]), count=n, offset=4)
            return np.stack([a["x"], a["y"], a["c"]], 1).astype(np.float64)
        xo, yo = r.i32(), r.i32()
        short_counts = r.i8() == 0
        short_x = short_y = True
        if self.version > 8:
            short_x = r.i8() == 0
            short_y = r.i8() == 0
        typ = r.i8()
        X = np.empty(n, np.int64); Y = np.empty(n, np.int64); C = np.empty(n, np.float64)
        k = 0
        if typ == 1:
            rx = r.i16 if short_x else r.i32
            ry = r.i16 if short_y else r.i32
            rc = r.i16 if short_counts else r.f32
            for _ in range(ry()):
                y = yo + ry()
                for _ in range(rx()):
                    X[k] = xo + rx(); Y[k] = y; C[k] = rc(); k += 1
        elif typ == 2:
            npts = r.i32()
            w = r.i16()
            dt = "<i2" if short_counts else "<f4"
            vals = np.frombuffer(buf, dt, count=npts, offset=r.p).astype(np.float64)
            idx = np.arange(npts)
            row, col = idx // w, idx % w
            ok = vals != -32768 if short_counts else ~np.isnan(vals)
            k = int(ok.sum())
            X[:k] = xo + col[ok]; Y[:k] = yo + row[ok]; C[:k] = vals[ok]
        else:
            raise ValueError(f"unknown .hic block type {typ}")
        return np.stack([X[:k], Y[:k], C[:k]], 1).astype(np.float64)
