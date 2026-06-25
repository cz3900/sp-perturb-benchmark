import numpy as np, h5py, pytest
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.spacetravlr_export import export_to_spacetravlr_h5

def _d():
    return StandardData(
        X=np.zeros((4, 3)), coords=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float),
        perturbation=np.array(["Gata3", CONTROL, UNLABELED, "Gata3"]),
        cell_type=np.array(["T", "T", "B", "B"]), batch=np.array(["s1", "s1", "s1", "s1"]),
        gene_names=["g1", "g2", "g3"], meta={})

def test_spacetravlr_export_fields(tmp_path):
    d = _d(); X = np.arange(12.).reshape(4, 3)
    info = export_to_spacetravlr_h5(d, X, str(tmp_path / "a.h5ad"), species="mouse")
    assert info["species"] == "mouse"
    with h5py.File(tmp_path / "a.h5ad", "r") as f:
        assert f["X"].shape == (4, 3)
        assert list(np.array(f["obs"]["cell_type"]).astype(str)) == ["T", "T", "B", "B"]
        assert np.allclose(np.array(f["obsm"]["spatial"]), d.coords)
        assert f.attrs["species"] == "mouse"

def test_spacetravlr_export_bad_species(tmp_path):
    with pytest.raises(ValueError):
        export_to_spacetravlr_h5(_d(), np.zeros((4, 3)), str(tmp_path / "a.h5ad"), species="rat")
