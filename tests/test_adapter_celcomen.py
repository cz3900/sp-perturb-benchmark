import numpy as np, h5py, pytest
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.celcomen_export import export_to_celcomen_h5

def _d(counts):
    return StandardData(
        X=np.zeros((4, 3)), coords=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float),
        perturbation=np.array(["GeneA", CONTROL, UNLABELED, "GeneA"]),
        cell_type=np.array(["T", "T", "B", "B"]), batch=np.array(["s1", "s1", "s1", "s1"]),
        gene_names=["g1", "g2", "g3"], meta={})

def test_celcomen_export_fields(tmp_path):
    counts = np.array([[2, 4, 0], [0, 6, 2], [4, 0, 8], [2, 2, 2]], float)
    d = _d(counts)
    export_to_celcomen_h5(d, counts, str(tmp_path / "a.h5ad"))
    with h5py.File(tmp_path / "a.h5ad", "r") as f:
        assert np.allclose(np.array(f["X"]), counts)
        assert list(np.array(f["obs"]["cell_type"]).astype(str)) == ["T", "T", "B", "B"]
        assert np.allclose(np.array(f["obsm"]["spatial"]), d.coords)

def test_celcomen_export_rejects_non_integer(tmp_path):
    d = _d(None)
    with pytest.raises(ValueError):
        export_to_celcomen_h5(d, np.full((4, 3), 0.5), str(tmp_path / "a.h5ad"))
