import numpy as np
import pytest

from hicscaf.evaluate import evaluate
from hicscaf.features import (ComponentSet, adjacencies_from_scaffolds, adjacencies_from_truth, build_graph,
                              junction_patches, map_contacts)
from hicscaf.io import Component, Scaffold, agp_to_fasta, read_agp, read_truth, revcomp, write_agp, write_truth
from hicscaf.scaffold import greedy_decode, prepare
from hicscaf.simulate import cut_reference_into_contigs, simulate


@pytest.fixture(scope="module")
def sim():
    return simulate(seed=7, chroms={"chrA": 2_000_000, "chrB": 1_500_000}, pairs_per_mb=10_000)


def test_truth_scores_perfect(sim):
    r = evaluate(sim.true_scaffolds, sim.truth)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["misjoins_total"] == 0


def test_draft_has_errors(sim):
    r = evaluate(sim.draft, sim.truth)
    assert r["precision"] < 1.0 or r["recall"] < 1.0


def test_reversed_scaffold_is_equivalent(sim):
    rev = [Scaffold(s.name, [(c, -st) for c, st in reversed(s.parts)]) for s in sim.true_scaffolds]
    assert evaluate(rev, sim.truth)["F1"] == 1.0


def test_agp_roundtrip(sim, tmp_path):
    write_agp(sim.draft, tmp_path / "d.agp")
    back = read_agp(tmp_path / "d.agp")
    assert [s.parts for s in back] == [s.parts for s in sim.draft]
    write_truth(sim.truth, tmp_path / "t.tsv")
    assert read_truth(tmp_path / "t.tsv") == sim.truth


def test_oracle_decode_recovers_truth(sim):
    cs = ComponentSet.from_scaffolds(sim.true_scaffolds)
    true = adjacencies_from_truth(cs, sim.truth)
    edges = np.array(sorted(true))
    probs = np.zeros((len(edges), 5))
    for i, k in enumerate(map(tuple, edges)):
        probs[i, true[k]] = 1
    out = greedy_decode(cs, edges, probs)
    assert len(out) == 2 and evaluate(out, sim.truth)["F1"] == 1.0


def test_contacts_land_inside_components(sim):
    cs = ComponentSet.from_scaffolds(sim.draft)
    a, pa, b, pb, _ = map_contacts(cs, sim.contacts)
    assert len(a) == len(sim.contacts)
    assert (pa < cs.lengths[a]).all() and (pb < cs.lengths[b]).all()


def test_true_neighbours_contact_more(sim):
    cs = ComponentSet.from_scaffolds(sim.draft)
    g = build_graph(cs, map_contacts(cs, sim.contacts), adjacencies_from_scaffolds(cs, sim.draft),
                    true_adj=adjacencies_from_truth(cs, sim.truth))
    pos = g.edge_y > 0
    assert pos.sum() > 0.8 * len(adjacencies_from_truth(cs, sim.truth))  # candidates cover the truth
    # log contact count at 100 kb scale, matching class column
    col = 20 + (g.edge_y[pos] - 1)
    assert g.edge_x[pos][np.arange(pos.sum()), col].mean() > g.edge_x[~pos][:, 20:24].max(1).mean()


def test_patches_shape(sim):
    cs, g, patches, keys, cls, bs, draft = prepare(sim.lengths, sim.contacts, sim.draft)
    assert patches.shape[1:] == (2, 48, 48) and len(keys) == len(cls) == len(patches)


def test_cut_reference_sequences():
    rng = np.random.default_rng(0)
    ref = {"c1": "".join(rng.choice(list("ACGT"), 200_000)), "c2": "".join(rng.choice(list("ACGT"), 120_000))}
    seqs, truth = cut_reference_into_contigs(ref, mean_len=30_000, min_len=5_000)
    for n, s in seqs.items():
        ch, b, e, st = truth[n]
        assert (s if st > 0 else revcomp(s)) == ref[ch][b:e]
    # rebuilding with the truth order gives back the reference
    parts = sorted(((truth[n][0], truth[n][1], n) for n in seqs))
    scafs = [Scaffold(ch, [(Component(n, 0, len(seqs[n])), truth[n][3]) for c, _, n in parts if c == ch]) for ch in ref]
    fa = agp_to_fasta(scafs, seqs, gap=0)
    assert fa == ref
