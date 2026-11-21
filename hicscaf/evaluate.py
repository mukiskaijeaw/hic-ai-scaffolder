"""Score scaffolds against a reference placement of the contigs."""
from __future__ import annotations

import numpy as np

from .features import ComponentSet, adjacencies_from_scaffolds, adjacencies_from_truth
from .io import Scaffold, component_truth


def nx(lengths, frac=0.5) -> int:
    s = sorted(lengths, reverse=True)
    tot, acc = sum(s), 0
    for l in s:
        acc += l
        if acc >= frac * tot:
            return l
    return 0


def evaluate(scafs: list[Scaffold], truth, relocation_dist: int = 1_000_000) -> dict:
    cs = ComponentSet.from_scaffolds(scafs)
    pred = adjacencies_from_scaffolds(cs, scafs)
    true = adjacencies_from_truth(cs, truth)
    tpos = {i: component_truth(c, truth) for i, c in enumerate(cs.comps)}
    correct = inversion = inter = reloc = local = unknown = 0
    for (a, b), k in pred.items():
        if true.get((a, b)) == k:
            correct += 1
            continue
        ta, tb = tpos[a], tpos[b]
        if ta is None or tb is None:
            unknown += 1
        elif ta[0] != tb[0]:
            inter += 1
        elif (a, b) in true:
            inversion += 1
        else:
            gap = max(ta[1], tb[1]) - min(ta[2], tb[2])
            if gap > relocation_dist:
                reloc += 1
            else:
                local += 1
    lens = [sum(c.length for c, _ in s.parts) for s in scafs]
    n_pred, n_true = len(pred), len(true)
    return {
        "scaffolds": len(scafs),
        "N50": nx(lens),
        "N90": nx(lens, 0.9),
        "largest": max(lens) if lens else 0,
        "joins": n_pred,
        "correct_joins": correct,
        "precision": correct / n_pred if n_pred else 1.0,
        "recall": correct / n_true if n_true else 1.0,
        "F1": 2 * correct / (n_pred + n_true) if (n_pred + n_true) else 1.0,
        "misjoin_interchrom": inter,
        "misjoin_relocation": reloc,
        "local_misorder": local,
        "orientation_error": inversion,
        "misjoins_total": inter + reloc,
        "unknown": unknown,
    }


COLUMNS = ["scaffolds", "N50", "largest", "joins", "precision", "recall", "F1", "misjoins_total",
           "misjoin_interchrom", "misjoin_relocation", "local_misorder", "orientation_error"]


def format_table(rows: dict[str, dict]) -> str:
    head = "| method | " + " | ".join(COLUMNS) + " |\n|" + "---|" * (len(COLUMNS) + 1) + "\n"
    body = ""
    for name, r in rows.items():
        vals = [f"{r[c]:.3f}" if isinstance(r[c], float) else f"{r[c]:,}" for c in COLUMNS]
        body += f"| {name} | " + " | ".join(vals) + " |\n"
    return head + body
