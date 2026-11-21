"""Grouping scaffolds into a known number of chromosomes."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
from hicscaf.evaluate import evaluate  # noqa: E402
from hicscaf.features import ComponentSet, map_contacts  # noqa: E402
from hicscaf.grouping import group_chromosomes  # noqa: E402
from hicscaf.io import component_truth, singleton_scaffolds  # noqa: E402
from hicscaf.simulate import simulate  # noqa: E402


def _purity(scafs, truth):
    out = []
    for s in scafs:
        if not s.name.startswith("chr"):
            continue
        d = {}
        for c, _ in s.parts:
            t = component_truth(c, truth)
            d[t[0]] = d.get(t[0], 0) + c.length
        out.append(max(d.values()) / sum(d.values()))
    return out


@pytest.mark.parametrize("seed", [0, 2])
def test_groups_unscaffolded_contigs(seed):
    """From single contigs only (no model, no draft): k groups, each from one true chromosome, in order."""
    rng = np.random.default_rng(seed)
    chroms = {f"c{i + 1}": int(rng.uniform(4e6, 9e6)) for i in range(5)}
    sim = simulate(seed=seed, chroms=chroms, pairs_per_mb=10_000, mean_contig=200_000)
    single = singleton_scaffolds(sim.lengths)
    cs = ComponentSet.from_scaffolds(single)
    out = group_chromosomes(cs, map_contacts(cs, sim.contacts), single, 5)
    groups = [s for s in out if s.name.startswith("chr")]
    assert len(groups) == 5
    assert min(_purity(out, sim.truth)) > 0.95
    placed = sum(c.length for s in groups for c, _ in s.parts) / sum(sim.lengths.values())
    assert placed > 0.95
    assert evaluate(out, sim.truth)["F1"] > 0.8


def test_zero_means_off():
    sim = simulate(seed=1)
    single = singleton_scaffolds(sim.lengths)
    cs = ComponentSet.from_scaffolds(single)
    assert group_chromosomes(cs, map_contacts(cs, sim.contacts), single, 0) is single
