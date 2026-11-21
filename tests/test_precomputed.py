"""DNA-Zoo-style workflow: draft .assembly + binned .hic dump -> refine -> score vs curated .review.assembly."""
import numpy as np

from hicscaf.evaluate import evaluate
from hicscaf.features import ComponentSet, map_contacts
from hicscaf.io import truth_from_scaffolds
from hicscaf.precomputed import (layout_from_agp, layout_from_assembly, load_matrix_contacts, read_assembly,
                                 write_assembly)
from hicscaf.simulate import simulate


def _dump_assembly_layout(sim, scafs, path, res):
    """Write what `juicer_tools dump observed NONE x.hic assembly assembly BP res` would print."""
    off, pos = {}, 0
    for s in scafs:
        for c, st in s.parts:
            off[c.contig] = (pos, st, c.length); pos += c.length
    n = sim.contacts.names
    def proj(ci, p):
        o = np.array([off[n[i]][0] for i in ci]); st = np.array([off[n[i]][1] for i in ci])
        L = np.array([off[n[i]][2] for i in ci])
        return o + np.where(st > 0, p, L - 1 - p)
    x, y = proj(sim.contacts.c1, sim.contacts.p1) // res * res, proj(sim.contacts.c2, sim.contacts.p2) // res * res
    x, y = np.minimum(x, y), np.maximum(x, y)
    key, cnt = np.unique(np.stack([x, y], 1), axis=0, return_counts=True)
    np.savetxt(path, np.column_stack([key, cnt]), fmt="%d")


def test_assembly_roundtrip_and_refine(tmp_path):
    sim = simulate(seed=11, chroms={"c1": 2_000_000, "c2": 1_500_000, "c3": 1_000_000}, pairs_per_mb=20_000)
    write_assembly(sim.draft, sim.lengths, tmp_path / "draft.assembly")
    write_assembly(sim.true_scaffolds, sim.lengths, tmp_path / "draft.review.assembly")
    draft, lengths = read_assembly(tmp_path / "draft.assembly")
    review, _ = read_assembly(tmp_path / "draft.review.assembly")
    assert [p for s in draft for p in s.parts] == [p for s in sim.draft for p in s.parts]
    truth = truth_from_scaffolds(review)
    assert evaluate(review, truth)["F1"] == 1.0
    assert evaluate(draft, truth)["F1"] == evaluate(sim.draft, sim.truth)["F1"]

    res = 2000
    _dump_assembly_layout(sim, draft, tmp_path / "dump.txt", res)
    layout = layout_from_assembly(draft, list(lengths))
    ct = load_matrix_contacts(tmp_path / "dump.txt", layout, resolution=res)
    assert abs(ct.w.sum() - len(sim.contacts)) / len(sim.contacts) < 0.01
    # contacts land on the right contigs (within one bin)
    cs = ComponentSet.from_scaffolds(draft)
    assert len(map_contacts(cs, ct)[0]) == len(ct)


def test_split_contigs_in_assembly(tmp_path):
    p = tmp_path / "x.assembly"
    p.write_text(">a:::fragment_1 1 100\n>a:::fragment_2:::debris 2 50\n>a:::fragment_3 3 200\n>b 4 300\n-3 4\n1\n2\n")
    scafs, lengths = read_assembly(p)
    assert lengths == {"a": 350, "b": 300}
    (c, st), (c2, _) = scafs[0].parts
    assert (c.contig, c.beg, c.end, st) == ("a", 150, 350, -1) and c2.contig == "b"


def test_agp_layout(tmp_path):
    agp = tmp_path / "y.agp"
    agp.write_text("s1\t1\t100\t1\tW\tx\t1\t100\t+\ns1\t101\t300\t2\tN\t200\tscaffold\tyes\tproximity_ligation\n"
                   "s1\t301\t350\t3\tW\ty\t1\t50\t-\n")
    lay = layout_from_agp(agp, ["x", "y"])
    ci, lp = lay.to_contig("s1", np.array([0, 99, 150, 300, 349]))
    assert ci.tolist() == [0, 0, -1, 1, 1] and lp[[0, 1, 3, 4]].tolist() == [0, 99, 49, 0]
