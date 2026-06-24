import numpy as np
import h5py
import pytest
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.scgen_export import build_lognorm_X, export_to_scgen_h5


def _toy():
    # 4 cells: GeneA(stim), control, none(unlabeled), GeneA(stim); integer counts in meta['counts']
    counts = np.array([[2, 4, 0], [0, 6, 2], [4, 0, 8], [2, 2, 2]], float)
    d = StandardData(
        X=np.zeros((4, 3), float),                       # X is z-scored / irrelevant here
        coords=np.array([[0, 0], [10, 0], [0, 10], [10, 10]], float),
        perturbation=np.array(["GeneA", CONTROL, UNLABELED, "GeneA"]),
        cell_type=np.array(["T", "T", "B", "B"]),
        batch=np.array(["s1"] * 4), gene_names=["g1", "g2", "g3"],
        meta={"counts": counts})
    return d


def test_build_lognorm_X_matches_recipe():
    d = _toy()
    X = build_lognorm_X(d)
    counts = d.meta["counts"]
    expected = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
    assert X.shape == (4, 3)
    assert np.allclose(X, expected)
    assert np.isfinite(X).all()


def test_build_lognorm_X_rejects_non_integer_counts():
    d = _toy()
    d.meta["counts"] = d.meta["counts"] + 0.5            # pre-normalized -> not raw integer counts
    with pytest.raises(ValueError):
        build_lognorm_X(d)


def test_export_to_scgen_h5_condition_map_and_drop(tmp_path):
    d = _toy()
    lognorm_X = build_lognorm_X(d)
    p = tmp_path / "GeneA.h5ad"
    info = export_to_scgen_h5(d, "GeneA", lognorm_X, str(p))
    assert info["n_stim"] == 2 and info["n_ctrl"] == 1
    with h5py.File(p, "r") as f:
        # kept = 2 GeneA (stim) + 1 control; 'none' dropped
        assert f["X"].shape == (3, 3)
        cond = list(np.array(f["obs"]["condition"]).astype(str))
        assert cond == ["stimulated", "control", "stimulated"]
        ct = list(np.array(f["obs"]["cell_type"]).astype(str))
        assert ct == ["T", "T", "B"]
        oi = list(np.array(f["obs"]["orig_idx"]))
        assert oi == [0, 1, 3]                            # orig cell ids of kept rows
        assert list(np.array(f["var"]["gene_names"]).astype(str)) == ["g1", "g2", "g3"]
        # normalization recipe stored for cross-env eval_X reproducibility
        assert float(f["uns"].attrs["target_sum"]) == 1e4
        assert int(f["uns"].attrs["log1p"]) == 1
        assert np.allclose(np.array(f["X"]), lognorm_X[[0, 1, 3]])
