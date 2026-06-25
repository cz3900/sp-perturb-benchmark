import numpy as np
import anndata as ad
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.spatialprop_export import export_to_spatialprop_h5ad
from spbench.adapters.counts_export import build_counts_X

def _d():
    counts = np.array([[2, 4, 0], [0, 6, 2], [4, 0, 8], [2, 2, 2]], float)
    return StandardData(
        X=np.zeros((4, 3)), coords=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float),
        perturbation=np.array(["GeneA", CONTROL, UNLABELED, "GeneA"]),
        cell_type=np.array(["T", "T", "B", "B"]), batch=np.array(["m1", "m1", "m2", "m2"]),
        gene_names=["g1", "g2", "g3"], meta={"counts": counts})

def test_spatialprop_export_real_anndata(tmp_path):
    d = _d(); X = build_counts_X(d)
    out = tmp_path / "all.h5ad"
    export_to_spatialprop_h5ad(d, X, str(out))
    a = ad.read_h5ad(out)                                      # the API uses sc.read_h5ad
    assert a.shape == (4, 3)
    assert np.allclose(np.asarray(a.X), X)                     # raw counts (model normalizes)
    assert list(a.obs["celltype"].astype(str)) == ["T", "T", "B", "B"]
    assert list(a.obs["mouse_id"].astype(str)) == ["m1", "m1", "m2", "m2"]
    assert np.allclose(np.asarray(a.obsm["spatial"]), d.coords)
    assert list(a.var_names) == ["g1", "g2", "g3"]
