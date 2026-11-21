"""hicscaf Studio backend: run a job end to end on simulated data (no browser)."""
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
from hicscaf import studio  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "models" / "hicscaf.pt"


@pytest.mark.skipif(not MODEL.exists(), reason="needs models/hicscaf.pt")
def test_job_end_to_end(tmp_path):
    files = studio.make_demo(tmp_path)
    job = studio.Job("t", {"mode": "correct", "min_prob": 0.5, "model": str(MODEL), **files}, tmp_path)
    studio.run_job(job)
    assert job.state == "done", job.error
    r = job.result
    assert r["after"]["scaffolds"] < r["before"]["scaffolds"]
    assert r["eval"]["after"]["F1"] > r["eval"]["before"]["F1"]
    assert (tmp_path / "corrected.assembly").exists() and (tmp_path / "corrected.agp").exists()
    assert (tmp_path / "corrected.hic").exists()
    m = studio.contact_map(job, "after", -1, 100)
    assert m["n"] == 100 and len(m["names"]) == r["after"]["scaffolds"]
    assert studio.scaffold_tracks(job, "after")["count"] == r["after"]["scaffolds"]
    studio.decode_and_write(job, 0.9)
    assert job.result["min_prob"] == 0.9
    json.loads((tmp_path / "summary.json").read_text())


@pytest.mark.skipif(not MODEL.exists(), reason="needs models/hicscaf.pt")
def test_job_groups_chromosomes(tmp_path):
    files = studio.make_demo(tmp_path)
    job = studio.Job("t", {"mode": "correct", "min_prob": 0.5, "model": str(MODEL), "n_chrom": 2, **files}, tmp_path)
    studio.run_job(job)
    assert job.state == "done", job.error
    ch = job.result["chroms"]
    assert len(ch["list"]) == 2 and ch["placed_frac"] > 0.9
    studio.decode_and_write(job, 0.5, 0)  # switch grouping off again from the results page
    assert job.result["chroms"] is None and job.params["n_chrom"] == 0
