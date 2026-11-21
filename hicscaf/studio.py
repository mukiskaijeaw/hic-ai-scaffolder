"""hicscaf Studio: a local web app around `hicscaf correct` / `hicscaf refine`.

    hicscaf studio            # opens http://127.0.0.1:8765 in the browser

Only the Python standard library is used for the server, so nothing extra has to be installed.
Uploaded files and results go to ~/hicscaf_studio (change with --workdir). Large files such as a
.hic can be used in place by pasting their path instead of uploading a copy.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import traceback
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

WEB = Path(__file__).with_name("studio_web")
REPO_MODEL = Path(__file__).resolve().parent.parent / "models" / "hicscaf.pt"

EXT = {
    "contigs": (".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz", ".fai", ".sizes", ".tsv"),
    "pairs": (".pairs", ".pairs.gz", ".bed", ".bed.gz", ".npz", ".txt", ".txt.gz"),
    "draft": (".agp", ".assembly"),
    "hic": (".hic", ".mcool", ".cool", ".txt"),
    "truth": (".tsv", ".agp", ".assembly"),
}
STEPS = [
    ("load", "อ่านไฟล์ Hi-C"),
    ("map", "จับคู่ contact กับ contig"),
    ("cnn", "JunctionCNN ตรวจรอยต่อ"),
    ("gnn", "EdgeGNN ทำนายคู่ปลาย contig"),
    ("decode", "Decoder ประกอบ scaffold ใหม่"),
    ("write", "เขียนไฟล์ผลลัพธ์"),
]


def _fmt_bp(x: float) -> str:
    return f"{x / 1e6:.2f} Mb" if x >= 1e6 else f"{x / 1e3:.1f} kb"


# ------------------------------------------------------------------- jobs
class Job:
    def __init__(self, jid: str, params: dict, outdir: Path):
        self.id, self.params, self.outdir = jid, params, outdir
        self.created = time.time()
        self.state = "queued"          # queued | running | done | error
        self._step = -1
        # per step: start / end time and fraction done (None = no estimate available)
        self.prog = [{"start": None, "end": None, "frac": None} for _ in STEPS]
        self.log: list[str] = []
        self.error = None
        self.lock = threading.Lock()
        self.ctx: dict = {}
        self.result: dict | None = None
        self.maps: dict = {}

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, i: int):
        now = time.time()
        with self.lock:
            for k in range(max(self._step, 0), min(i, len(STEPS))):
                if self.prog[k]["start"] is not None and self.prog[k]["end"] is None:
                    self.prog[k]["end"] = now
                    self.prog[k]["frac"] = 1.0
            if 0 <= i < len(STEPS) and self.prog[i]["start"] is None:
                self.prog[i]["start"] = now
                self.prog[i]["frac"] = 0.0
            self._step = i

    def progress(self, frac: float):
        """Report progress of the current step (0..1)."""
        if 0 <= self._step < len(STEPS):
            self.prog[self._step]["frac"] = float(min(1.0, max(0.0, frac)))

    def say(self, msg: str):
        t = time.time() - self.created
        with self.lock:
            self.log.append(f"[{t:6.1f}s] {msg}")

    def to_status(self):
        now = time.time()
        with self.lock:
            steps = []
            for (key, label), pr in zip(STEPS, self.prog):
                st, en, fr = pr["start"], pr["end"], pr["frac"]
                el = (en or now) - st if st else 0.0
                eta = None
                if st and not en and fr is not None and fr >= 0.01 and el >= 2:
                    eta = el * (1 - fr) / fr
                steps.append({"key": key, "label": label, "elapsed": round(el, 1), "frac": fr,
                              "eta": None if eta is None else round(eta, 1), "done": en is not None})
            return {"id": self.id, "state": self.state, "step": self._step, "steps": [s[1] for s in STEPS],
                    "detail": steps, "log": self.log[-200:], "error": self.error, "params": self.params,
                    "created": self.created, "now": now, "name": self.params.get("name") or self.id}


JOBS: dict[str, Job] = {}


def _resolve(p: str | None) -> str | None:
    if not p:
        return None
    p = p.strip().strip('"').strip("'")
    if not os.path.exists(p):
        raise ValueError(f"ไม่พบไฟล์: {p}")
    return p


def _cache_key(path: str, extra: str) -> str:
    import hashlib
    st = os.stat(path)
    return hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{extra}".encode()).hexdigest()[:16]


def _score_junctions(job, cnn, patches, batch=256):
    if cnn is None or len(patches) == 0:
        return np.full(len(patches), 0.5)
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(patches), batch):
            out.append(torch.sigmoid(cnn(torch.from_numpy(patches[i:i + batch]))).numpy())
            job.progress(0.5 + 0.5 * min(1.0, (i + batch) / len(patches)))
    return np.concatenate(out)


def run_job(job: Job):
    from .features import adjacencies_from_scaffolds, auto_bin_size, build_graph, junction_patches, map_contacts, \
        ComponentSet
    from .io import Contacts, add_missing_components, read_lengths, read_pairs
    from .models import load
    from .precomputed import layout_from_agp, layout_from_assembly, load_matrix_contacts, load_scaffolds
    from .scaffold import edge_probs

    P = job.params
    cache_dir = job.outdir.parent.parent / "cache"
    try:
        job.state = "running"
        # 1 read files ------------------------------------------------------------
        job.step = 0
        contigs, pairs, draft_p, hic, truth_p = (_resolve(P.get(k)) for k in ("contigs", "pairs", "draft", "hic", "truth"))
        model = _resolve(P.get("model") or str(REPO_MODEL))
        draft, lengths = (load_scaffolds(draft_p) if draft_p else (None, {}))
        if contigs:
            job.say(f"อ่านความยาว contig จาก {Path(contigs).name}")
            lengths = read_lengths(contigs)
            job.say(f"contig {len(lengths):,} ตัว รวม {_fmt_bp(sum(lengths.values()))}")
        if draft is not None:
            job.say(f"draft: {len(draft):,} scaffold จาก {Path(draft_p).name}")
        src = hic if P["mode"] == "refine" else pairs
        if not src:
            raise ValueError("โหมด .hic ต้องมีไฟล์ contact map และ draft (.assembly / .agp)" if P["mode"] == "refine"
                             else "โหมด pairs ต้องมีไฟล์ contigs และไฟล์ Hi-C pairs")
        res = int(P.get("resolution") or 5000)
        extra = f"{P['mode']}|{res}|{P.get('scale')}|{_cache_key(draft_p, '') if (draft_p and P['mode'] == 'refine') else ''}"
        cache = cache_dir / f"{_cache_key(src, extra)}.npz"
        if cache.exists() and not str(src).endswith(".npz"):
            job.say(f"ใช้ข้อมูลที่อ่านไว้แล้วจากรอบก่อน ({Path(src).name}) ไม่ต้องอ่านไฟล์ใหม่")
            contacts = Contacts.load(cache)
        elif P["mode"] == "refine":
            if not draft_p:
                raise ValueError("โหมด .hic ต้องมีไฟล์ draft (.assembly / .agp)")
            names = list(lengths)
            layout = layout_from_assembly(draft, names) if draft_p.endswith(".assembly") else layout_from_agp(draft_p, names)
            job.say(f"อ่าน {Path(hic).name} ที่ความละเอียด {res:,} bp ({os.path.getsize(hic) / 1e9:.1f} GB)")
            contacts = load_matrix_contacts(hic, layout, res, P.get("juicer_tools") or None, int(P.get("scale") or 1),
                                            progress=job.progress)
        else:
            if not contigs:
                raise ValueError("โหมด pairs ต้องมีไฟล์ contigs (.fa / .fai)")
            job.say(f"อ่าน Hi-C pairs จาก {Path(pairs).name} ({os.path.getsize(pairs) / 1e9:.2f} GB)")
            contacts = Contacts.load(pairs) if pairs.endswith(".npz") else read_pairs(pairs, progress=job.progress)
        if not cache.exists() and not str(src).endswith(".npz"):
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(".tmp.npz")
                np.savez(tmp, names=np.array(contacts.names), c1=contacts.c1, p1=contacts.p1, c2=contacts.c2,
                         p2=contacts.p2, w=contacts.w, resolution=contacts.resolution)  # uncompressed: fast
                os.replace(tmp, cache)
                job.say("บันทึกข้อมูลที่อ่านแล้วไว้ รอบหน้าจะเร็วขึ้น")
            except OSError:
                pass
        job.say(f"contact {float(contacts.w.sum()):,.0f} คู่ ({len(contacts):,} แถว)")
        if not lengths:
            raise ValueError("ไม่รู้ความยาว contig: ใส่ไฟล์ contigs (.fa / .fai)")

        # 2 map -----------------------------------------------------------------
        job.step = 1
        cnn, gnn, meta = load(model)
        job.progress(0.1)
        draft2 = add_missing_components(draft or [], lengths)
        cs = ComponentSet.from_scaffolds(draft2)
        job.progress(0.2)
        mapped = map_contacts(cs, contacts)
        del contacts
        job.progress(0.9)
        bin_size = int(P.get("bin_size") or 0) or auto_bin_size(
            mapped, int(cs.lengths.sum()), min_bin=max(1_000, 2 * job_resolution(P)))
        job.say(f"ชิ้น contig {len(cs.comps):,} ชิ้น, contact ที่ใช้ได้ {float(mapped[4].sum()):,.0f}, bin {bin_size:,} bp")

        # 3 cnn -----------------------------------------------------------------
        job.step = 2
        dadj = adjacencies_from_scaffolds(cs, draft2)
        patches, jkeys, _ = junction_patches(cs, draft2, mapped, bin_size, progress=lambda f: job.progress(0.5 * f))
        js = _score_junctions(job, cnn, patches)
        jscore = dict(zip(jkeys, js))
        job.say(f"ตรวจรอยต่อ {len(jkeys):,} จุด: น่าสงสัย (P<0.5) {int((js < 0.5).sum()):,} จุด")

        # 4 gnn -----------------------------------------------------------------
        job.step = 3
        job.prog[3]["frac"] = None  # one big call: no progress available
        g = build_graph(cs, mapped, dadj, jscore)
        job.progress(0.6)
        probs = edge_probs(gnn, g)
        job.say(f"graph {len(cs.comps):,} node, {len(g.edge_index):,} edge")

        job.ctx = dict(cs=cs, draft=draft2, mapped=mapped, edges=g.edge_index, probs=probs, jscore=jscore,
                       dadj=dadj, lengths=lengths, contigs=contigs, truth=truth_p, bin_size=bin_size,
                       fasta=contigs if contigs and not contigs.endswith((".fai", ".sizes", ".tsv")) else None)
        # 5-6 decode + write ------------------------------------------------------
        job.step = 4
        decode_and_write(job, float(P.get("min_prob") or 0.5))
        job.step = len(STEPS)
        job.state = "done"
        job.say("เสร็จสิ้น")
    except Exception as e:  # report to the UI
        job.state = "error"
        job.error = str(e) or e.__class__.__name__
        job.say("ผิดพลาด: " + job.error)
        traceback.print_exc()


def job_resolution(P) -> int:
    return int(P.get("resolution") or 5000) * int(P.get("scale") or 1) if P.get("mode") == "refine" else 0


def decode_and_write(job: Job, min_prob: float):
    from .evaluate import evaluate, nx
    from .features import adjacencies_from_scaffolds
    from .io import agp_to_fasta, read_fasta, write_agp, write_fasta
    from .precomputed import write_assembly
    from .scaffold import greedy_decode

    c = job.ctx
    cs, draft, edges, probs = c["cs"], c["draft"], c["edges"], c["probs"]
    live = job.state == "running"
    scafs = greedy_decode(cs, edges, probs, min_prob)
    if live:
        job.step = 5
    out = job.outdir
    write_agp(scafs, out / "corrected.agp")
    write_assembly(scafs, c["lengths"], out / "corrected.assembly")
    if live:
        job.progress(0.15)
    if c["fasta"]:
        if "seqs" not in c:
            job.say("อ่านลำดับเบสเพื่อเขียน corrected.fa")
            c["seqs"] = read_fasta(c["fasta"])
        if live:
            job.progress(0.5)
        write_fasta(agp_to_fasta(scafs, c["seqs"]), out / "corrected.fa")
    if live:
        job.progress(0.85)
    edits = compute_edits(cs, draft, scafs, edges, probs, c["jscore"], c["dadj"], min_prob)
    with open(out / "edits.tsv", "w", encoding="utf-8") as fh:
        fh.write("type\tcontigs\tdetail\tgnn_prob\tcnn_prob\tstatus\n")
        for e in edits:
            fh.write(f"{e['type']}\t{','.join(e['contigs'])}\t{e['detail']}\t{e['prob']}\t{e['cnn']}\t{e['status']}\n")

    def summary(sc):
        lens = [sum(p.length for p, _ in s.parts) for s in sc]
        return {"scaffolds": len(sc), "N50": nx(lens), "N90": nx(lens, 0.9), "largest": max(lens) if lens else 0,
                "total": int(sum(lens)), "joins": sum(len(s.parts) - 1 for s in sc)}

    flips = sum(1 for e in edits if e["type"] == "flip" and e["status"] == "applied")
    res = {"min_prob": min_prob, "before": summary(draft), "after": summary(scafs),
           "counts": {k: sum(1 for e in edits if e["type"] == k and e["status"] == "applied")
                      for k in ("break", "join", "flip")},
           "review": sum(1 for e in edits if e["status"] == "review"), "flipped": flips,
           "edits": edits, "files": [], "eval": None, "bin_size": c["bin_size"],
           "components": len(cs.comps)}
    if c["truth"]:
        from .cli import _load_truth
        t = _load_truth(c["truth"])
        res["eval"] = {"before": evaluate(draft, t), "after": evaluate(scafs, t)}
    for f in ("corrected.assembly", "corrected.agp", "corrected.fa", "edits.tsv"):
        if (out / f).exists():
            res["files"].append({"name": f, "size": (out / f).stat().st_size})
    (out / "summary.json").write_text(json.dumps({k: v for k, v in res.items() if k != "edits"}, indent=2))
    c["scafs"] = scafs
    job.maps = {}
    job.result = res
    job.say(f"เกณฑ์ {min_prob:.2f}: {res['before']['scaffolds']:,} → {res['after']['scaffolds']:,} scaffold, "
            f"N50 {_fmt_bp(res['before']['N50'])} → {_fmt_bp(res['after']['N50'])}")


def compute_edits(cs, draft, scafs, edges, probs, jscore, dadj, min_prob):
    """Differences between the draft and the corrected scaffolding, plus near-threshold proposals."""
    from .features import adjacencies_from_scaffolds

    cadj = adjacencies_from_scaffolds(cs, scafs)
    eidx = {(int(a), int(b)): i for i, (a, b) in enumerate(edges)}
    dscaf = {}
    for si, s in enumerate(draft):
        for c, st in s.parts:
            dscaf[cs.index[c]] = (si, st)
    name = lambda i: cs.comps[i].contig if (cs.comps[i].beg == 0) else cs.comps[i].name
    dname = lambda i: draft[dscaf[i][0]].name
    pr = lambda key, k: float(probs[eidx[key], k]) if key in eidx else None
    out = []
    for key, k in dadj.items():
        if cadj.get(key) == k:
            continue
        typ = "flip" if key in cadj else "break"
        cnn = jscore.get(key)
        out.append({"type": typ, "a": key[0], "b": key[1], "contigs": [name(key[0]), name(key[1])],
                    "detail": (f"เปลี่ยนทิศรอยต่อใน {dname(key[0])}" if typ == "flip"
                               else f"ตัดรอยต่อใน {dname(key[0])}"),
                    "prob": round(pr(key, cadj[key]) if typ == "flip" else 1 - (pr(key, k) or 0), 3),
                    "cnn": None if cnn is None else round(float(cnn), 3), "status": "applied"})
    for key, k in cadj.items():
        if key in dadj:
            continue
        same = dscaf[key[0]][0] == dscaf[key[1]][0]
        out.append({"type": "join", "a": key[0], "b": key[1], "contigs": [name(key[0]), name(key[1])],
                    "detail": (f"จัดลำดับใหม่ภายใน {dname(key[0])}" if same
                               else f"ต่อ {dname(key[0])} กับ {dname(key[1])}"),
                    "prob": round(pr(key, k) or 0, 3), "cnn": None, "status": "applied"})
    # proposals just below the threshold: worth checking in Juicebox
    lo = max(0.2, min_prob - 0.25)
    for i, (a, b) in enumerate(edges):
        key = (int(a), int(b))
        if key in cadj or key in dadj:
            continue
        k = int(np.argmax(probs[i, 1:])) + 1
        p = float(probs[i, k])
        if lo <= p < min_prob:
            out.append({"type": "join", "a": key[0], "b": key[1], "contigs": [name(key[0]), name(key[1])],
                        "detail": f"เสนอต่อ {dname(key[0])} กับ {dname(key[1])}", "prob": round(p, 3), "cnn": None,
                        "status": "review"})
    for key, k in dadj.items():  # draft junctions kept, but the model is unsure
        if cadj.get(key) == k:
            p = pr(key, k)
            if p is not None and p < min_prob + 0.1:
                out.append({"type": "keep", "a": key[0], "b": key[1], "contigs": [name(key[0]), name(key[1])],
                            "detail": f"รอยต่อเดิมใน {dname(key[0])} ที่ไม่แน่ใจ", "prob": round(p, 3),
                            "cnn": None if jscore.get(key) is None else round(float(jscore[key]), 3),
                            "status": "review"})
    out.sort(key=lambda e: (e["status"] != "applied", -e["prob"]))
    for i, e in enumerate(out):
        e["id"] = i
    return out


# ------------------------------------------------------------------- maps
def contact_map(job: Job, which: str, scaf: int, n: int) -> dict:
    key = (which, scaf, n)
    if key in job.maps:
        return job.maps[key]
    c = job.ctx
    cs = c["cs"]
    sc = c["draft"] if which == "before" else c["scafs"]
    order = sorted(range(len(sc)), key=lambda i: -sum(p.length for p, _ in sc[i].parts))
    if scaf >= 0:
        order = [order[scaf]] if scaf < len(order) else []
    N = len(cs.comps)
    off = np.full(N, -1, np.int64)
    strand = np.ones(N, np.int64)
    bounds, names, comp_rows = [], [], []
    pos = 0
    for si in order:
        s = sc[si]
        bounds.append(pos); names.append(s.name)
        row = []
        for comp, st in s.parts:
            i = cs.index[comp]
            off[i], strand[i] = pos, st
            row.append([i, int(comp.length), int(st)])
            pos += comp.length
        comp_rows.append(row)
    total = max(pos, 1)
    a, pa, b, pb, w = c["mapped"]
    L = cs.lengths
    m = (off[a] >= 0) & (off[b] >= 0)
    a, pa, b, pb, w = a[m], pa[m], b[m], pb[m], w[m]
    x = off[a] + np.where(strand[a] > 0, pa, L[a] - 1 - pa)
    y = off[b] + np.where(strand[b] > 0, pb, L[b] - 1 - pb)
    i = np.minimum((x * n) // total, n - 1)
    j = np.minimum((y * n) // total, n - 1)
    H = np.bincount(i * n + j, weights=w, minlength=n * n).reshape(n, n)
    H = H + H.T
    v = np.log1p(H)
    nz = v[v > 0]
    vmax = float(np.quantile(nz, 0.995)) if len(nz) else 1.0
    img = np.clip(v / max(vmax, 1e-9) * 255, 0, 255).astype(np.uint8)
    out = {"n": n, "total": int(total), "bounds": [int(b_) for b_ in bounds], "names": names,
           "data": base64.b64encode(img.tobytes()).decode()}
    if scaf >= 0 and order:
        out["parts"] = comp_rows[0]
    job.maps[key] = out
    return out


def scaffold_tracks(job: Job, which: str, limit: int = 60) -> dict:
    """Per scaffold: list of (component, length, strand, draft-scaffold index, flipped vs draft)."""
    c = job.ctx
    cs = c["cs"]
    draft = c["draft"]
    sc = draft if which == "before" else c["scafs"]
    dpos = {}
    for si, s in enumerate(draft):
        for k, (comp, st) in enumerate(s.parts):
            dpos[cs.index[comp]] = (si, st, k)
    dorder = {si: r for r, si in enumerate(sorted(range(len(draft)), key=lambda i: -sum(p.length for p, _ in draft[i].parts)))}
    order = sorted(range(len(sc)), key=lambda i: -sum(p.length for p, _ in sc[i].parts))
    rows = []
    for si in order[:limit]:
        s = sc[si]
        parts = [(cs.index[comp], comp.length, st) for comp, st in s.parts]
        # orientation of this scaffold relative to the draft scaffold it mostly came from
        votes: dict = {}
        for i, l, st in parts:
            d, dst, _ = dpos[i]
            votes.setdefault(d, [0, 0])
            votes[d][0 if st * dst > 0 else 1] += l
        rel = {d: (1 if v[0] >= v[1] else -1) for d, v in votes.items()}
        main = max(votes, key=lambda d: sum(votes[d]))
        rows.append({"name": s.name, "len": int(sum(l for _, l, _ in parts)), "parts": [
            {"i": int(i), "len": int(l), "st": int(st), "d": int(dorder[dpos[i][0]]),
             "flip": bool(st * dpos[i][1] != rel[dpos[i][0]]), "foreign": dpos[i][0] != main,
             "name": cs.comps[i].contig} for i, l, st in parts]})
    return {"rows": rows, "more": max(0, len(sc) - limit), "count": len(sc)}


# ------------------------------------------------------------------- demo
def make_demo(workdir: Path) -> dict:
    from .io import write_agp, write_truth
    from .simulate import simulate

    d = workdir / "demo"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "contacts.npz").exists():
        sim = simulate(seed=7, pairs_per_mb=20_000)
        with open(d / "contigs.sizes", "w") as fh:
            fh.writelines(f"{k}\t{v}\n" for k, v in sim.lengths.items())
        sim.contacts.save(d / "contacts.npz")
        write_agp(sim.draft, d / "draft.agp")
        write_truth(sim.truth, d / "truth.tsv")
    return {"contigs": str(d / "contigs.sizes"), "pairs": str(d / "contacts.npz"), "draft": str(d / "draft.agp"),
            "truth": str(d / "truth.tsv")}


def env_info() -> dict:
    def has(m):
        try:
            __import__(m)
            return True
        except Exception:
            return False
    return {"hicstraw": has("hicstraw"), "cooler": has("cooler"), "java": bool(shutil.which("java")),
            "torch": has("torch"), "model": str(REPO_MODEL) if REPO_MODEL.exists() else None,
            "juicer_tools": os.environ.get("JUICER_TOOLS")}


# ----------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    workdir: Path = Path.home() / "hicscaf_studio"
    server_version = "hicscaf-studio"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/jobs/" in (args[0] if args else "") and "GET" in (args[0] if args else ""):
            return
        super().log_message(fmt, *args)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg, code=400):
        self._json({"error": msg}, code)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _file(self, path: Path, download=False):
        if not path.is_file():
            return self._err("not found", 404)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".html", ".js", ".css"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype if not download else "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with open(path, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile, 1 << 20)

    def _job(self, jid):
        j = JOBS.get(jid)
        if not j:
            self._err("ไม่พบงานนี้ (โปรแกรมอาจถูกปิดแล้วเปิดใหม่)", 404)
        return j

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
        try:
            if not parts or parts[0] in ("index.html",):
                return self._file(WEB / "index.html")
            if parts[0] != "api":
                return self._file(WEB / Path(*parts).name)
            if parts[1] == "env":
                return self._json(env_info())
            if parts[1] == "jobs" and len(parts) == 2:
                return self._json([j.to_status() for j in sorted(JOBS.values(), key=lambda j: -j.created)])
            if parts[1] == "jobs":
                j = self._job(parts[2])
                if not j:
                    return
                if len(parts) == 3:
                    return self._json(j.to_status())
                if j.state != "done":
                    return self._err("งานยังไม่เสร็จ", 409)
                if parts[3] == "result":
                    return self._json(j.result)
                if parts[3] == "map":
                    with j.lock:
                        return self._json(contact_map(j, q.get("which", "after"), int(q.get("scaf", -1)),
                                                      min(1024, int(q.get("n", 600)))))
                if parts[3] == "tracks":
                    return self._json(scaffold_tracks(j, q.get("which", "after")))
                if parts[3] == "file":
                    name = Path(parts[4]).name
                    return self._file(j.outdir / name, download=True)
            self._err("not found", 404)
        except Exception as e:
            traceback.print_exc()
            self._err(str(e), 500)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = [unquote(p) for p in u.path.strip("/").split("/") if p]
        try:
            if parts[:2] == ["api", "upload"]:
                kind = q.get("kind", "")
                name = re.sub(r"[^\w.\-]+", "_", Path(q.get("name", "upload.bin")).name)
                if kind in EXT and not name.lower().endswith(EXT[kind]):
                    return self._err(f"ชนิดไฟล์ไม่ถูกต้อง: ต้องเป็น {', '.join(EXT[kind])}")
                d = self.workdir / "uploads" / uuid.uuid4().hex[:8]
                d.mkdir(parents=True, exist_ok=True)
                n = int(self.headers.get("Content-Length") or 0)
                with open(d / name, "wb") as fh:
                    left = n
                    while left > 0:
                        chunk = self.rfile.read(min(left, 1 << 20))
                        if not chunk:
                            break
                        fh.write(chunk)
                        left -= len(chunk)
                return self._json({"path": str(d / name), "name": name, "size": n})
            if parts[:2] == ["api", "check"]:
                b = self._body()
                p = (b.get("path") or "").strip().strip('"')
                kind = b.get("kind", "")
                if not os.path.isfile(p):
                    return self._err(f"ไม่พบไฟล์ที่ {p}")
                if kind in EXT and not p.lower().endswith(EXT[kind]):
                    return self._err(f"ชนิดไฟล์ไม่ถูกต้อง: ต้องเป็น {', '.join(EXT[kind])}")
                return self._json({"path": p, "name": Path(p).name, "size": os.path.getsize(p)})
            if parts[:2] == ["api", "demo"]:
                return self._json(make_demo(self.workdir))
            if parts[:2] == ["api", "jobs"] and len(parts) == 2:
                b = self._body()
                jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
                outdir = self.workdir / "jobs" / jid
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / "params.json").write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding="utf-8")
                job = Job(jid, b, outdir)
                JOBS[jid] = job
                threading.Thread(target=run_job, args=(job,), daemon=True).start()
                return self._json(job.to_status())
            if parts[:2] == ["api", "jobs"] and len(parts) == 4 and parts[3] == "threshold":
                j = self._job(parts[2])
                if not j:
                    return
                mp = float(self._body().get("min_prob", 0.5))
                decode_and_write(j, min(0.99, max(0.05, mp)))
                return self._json(j.result)
            self._err("not found", 404)
        except Exception as e:
            traceback.print_exc()
            self._err(str(e), 500)


class StudioServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets bind() succeed on a port another program already uses,
    # so the free-port probe below would silently shadow it. Use exclusive binding there.
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        import socket
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(host="127.0.0.1", port=8765, workdir=None, open_browser=True):
    if workdir:
        Handler.workdir = Path(workdir)
    Handler.workdir.mkdir(parents=True, exist_ok=True)
    srv = None
    for p in range(port, port + 20):
        try:
            srv = StudioServer((host, p), Handler)
            port = p
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit(f"ไม่มี port ว่างระหว่าง {port}-{port + 19}")
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/"
    print(f"hicscaf Studio พร้อมใช้งานที่ {url}", flush=True)
    print(f"ไฟล์งานเก็บที่ {Handler.workdir}", flush=True)
    print("ปิดหน้าต่างนี้ (หรือกด Ctrl+C) เพื่อหยุดโปรแกรม", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
