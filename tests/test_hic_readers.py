"""The built-in .hic reader and hictkpy must return the same counts."""
import numpy as np
import pytest

from hicscaf.hicfile import HicFile

hictkpy = pytest.importorskip("hictkpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")  # hictkpy needs it to write the fixture


@pytest.fixture(scope="module")
def hic_path(tmp_path_factory):
    rng = np.random.default_rng(0)
    chroms = {"chrA": 3_000_000, "chrB": 1_700_000}
    res = 10_000
    nA, nB = -(-chroms["chrA"] // res), -(-chroms["chrB"] // res)
    n = nA + nB
    b1 = rng.integers(0, n, 20_000)
    b2 = np.minimum(n - 1, b1 + rng.geometric(0.05, len(b1)) - 1)
    df = pd.DataFrame({"bin1_id": b1, "bin2_id": b2, "count": rng.integers(1, 400, len(b1))})
    df = df.groupby(["bin1_id", "bin2_id"], as_index=False)["count"].sum()
    p = tmp_path_factory.mktemp("hic") / "t.hic"
    w = hictkpy.hic.FileWriter(str(p), chroms, [res, 50_000], tmpdir=str(p.parent))
    w.add_pixels(df)
    w.finalize()
    return p, res


def _as_set(r):
    return {(int(x), int(y), float(c)) for x, y, c in r}


def test_builtin_matches_hictk(hic_path):
    from hicscaf.precomputed import _records_hictk

    p, res = hic_path
    h = HicFile.open(p)
    assert res in h.resolutions
    names = {c.name: c for c in h.chroms}
    ours = {}
    for i, a in enumerate(["chrA", "chrB"]):
        for b in ["chrA", "chrB"][i:]:
            ours[(a, b)] = _as_set(h.records(names[a], names[b], res))
    theirs = {(a, b): _as_set(r) for a, b, r in _records_hictk(p, res)}
    assert set(theirs) <= set(ours)
    for k, v in ours.items():
        assert v == theirs.get(k, set()), k
    assert sum(len(v) for v in ours.values()) > 1000
