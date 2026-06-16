import numpy as np, h5py
from spbench.adapters.saunders import SaundersAdapter

def _write_fake_h5mu(path):
    """Mirror the real Saunders structure: mod/rna with layers/raw_scaled, obs/singlet_gene
    (categorical with '' and 'control'), obs/cell_type, obsm/spatial."""
    n, g = 12, 4
    with h5py.File(path, "w") as h:
        rna = h.create_group("mod/rna")
        rna.create_dataset("layers/raw_scaled", data=np.random.rand(n, g))
        rna.create_dataset("obsm/spatial", data=np.random.rand(n, 2))
        rna.create_dataset("var/_index", data=np.array([f"g{i}".encode() for i in range(g)]))
        sg = rna.create_group("obs/singlet_gene")
        cats = np.array([b"", b"control", b"Mdm2"], dtype="S")
        sg.create_dataset("categories", data=cats)
        sg.create_dataset("codes", data=np.array([0, 1, 2, 0, 0, 1, 2, 0, 0, 0, 2, 1], dtype="i1"))
        ct = rna.create_group("obs/cell_type")
        ct.create_dataset("categories", data=np.array([b"Hep", b"Endo"], dtype="S"))
        ct.create_dataset("codes", data=np.zeros(n, dtype="i1"))

def test_normalizes_to_standarddata(tmp_path):
    p = tmp_path / "Batch_0_Slice_0.h5mu"
    _write_fake_h5mu(str(p))
    d = SaundersAdapter(str(tmp_path)).load()
    assert d.n_genes == 4
    assert set(np.unique(d.perturbation)) <= {"none", "control", "Mdm2"}
    assert "none" in d.perturbation
    assert d.is_control.sum() > 0
    assert d.coords.shape[1] == 2
