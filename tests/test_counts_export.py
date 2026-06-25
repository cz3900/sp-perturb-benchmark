import numpy as np, h5py, pytest
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.counts_export import build_counts_X, export_counts_h5

def _toy():
    counts = np.array([[2, 4, 0], [0, 6, 2], [4, 0, 8], [2, 2, 2]], float)
    return StandardData(
        X=np.zeros((4, 3)), coords=np.array([[0, 0], [10, 0], [0, 10], [10, 10]], float),
        perturbation=np.array(["GeneA", CONTROL, UNLABELED, "GeneA"]),
        cell_type=np.array(["T", "T", "B", "B"]), batch=np.array(["s1"] * 4),
        gene_names=["g1", "g2", "g3"], meta={"counts": counts})

def test_build_counts_X_passthrough_and_integer_guard():
    d = _toy()
    assert np.allclose(build_counts_X(d), d.meta["counts"])
    d.meta["counts"] = d.meta["counts"] + 0.5
    with pytest.raises(ValueError):
        build_counts_X(d)

def test_export_counts_h5_configurable_keys(tmp_path):
    d = _toy(); X = build_counts_X(d)
    p = tmp_path / "GeneA.h5ad"
    info = export_counts_h5(d, "GeneA", X, str(p),
                            stim_cond="GeneA+ctrl", ctrl_cond="ctrl",
                            cell_type_key="celltype", gene_key="gene_name")
    assert info["n_stim"] == 2 and info["n_ctrl"] == 1
    with h5py.File(p, "r") as f:
        assert f["X"].shape == (3, 3)                                  # 2 stim + 1 ctrl, 'none' dropped
        assert np.allclose(np.array(f["X"]), X[[0, 1, 3]])             # raw counts preserved
        cond = list(np.array(f["obs"]["condition"]).astype(str))
        assert cond == ["GeneA+ctrl", "ctrl", "GeneA+ctrl"]
        assert list(np.array(f["obs"]["celltype"]).astype(str)) == ["T", "T", "B"]
        assert list(np.array(f["obs"]["orig_idx"])) == [0, 1, 3]
        assert list(np.array(f["var"]["gene_name"]).astype(str)) == ["g1", "g2", "g3"]
