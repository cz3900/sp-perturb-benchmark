import numpy as np, h5py
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.spatialprop_export import export_to_spatialprop_h5
from spbench.adapters.counts_export import build_counts_X

def _d():
    counts = np.array([[2, 4, 0], [0, 6, 2], [4, 0, 8], [2, 2, 2]], float)
    return StandardData(
        X=np.zeros((4, 3)), coords=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float),
        perturbation=np.array(["GeneA", CONTROL, UNLABELED, "GeneA"]),
        cell_type=np.array(["T", "T", "B", "B"]), batch=np.array(["m1", "m1", "m2", "m2"]),
        gene_names=["g1", "g2", "g3"], meta={"counts": counts})

def test_spatialprop_export_required_obs_obsm_fields(tmp_path):
    d = _d(); X = build_counts_X(d)
    export_to_spatialprop_h5(d, X, str(tmp_path / "all.h5ad"))
    with h5py.File(tmp_path / "all.h5ad", "r") as f:
        assert f["X"].shape == (4, 3)
        assert np.allclose(np.array(f["X"]), X)                  # raw counts (model normalizes)
        assert list(np.array(f["obs"]["mouse_id"]).astype(str)) == ["m1", "m1", "m2", "m2"]
        assert list(np.array(f["obs"]["celltype"]).astype(str)) == ["T", "T", "B", "B"]
        assert np.allclose(np.array(f["obsm"]["spatial"]), d.coords)
