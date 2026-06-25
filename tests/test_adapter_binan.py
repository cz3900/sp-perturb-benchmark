import numpy as np
from spbench.adapters.binan import _assemble_binan


def test_binan_assemble_named_guides():
    X = np.arange(6 * 3).reshape(6, 3).astype(float)
    coords = np.arange(12).reshape(6, 2).astype(float)
    genes = ["g0", "g1", "g2"]
    # tumor cells at 1-based full index 2, 4, 5; named guides from the tumor perturbation table
    tumor_pert = {2: "CD14", 4: "control", 5: "CHUK"}
    data = _assemble_binan(X, coords, genes, tumor_pert,
                           tumor_idx={2, 4, 5}, immune_nb_idx={2}, without_nb_idx={4, 5})
    # full idx -> row (idx-1); tumor cells get their named guide / 'control'; everyone else 'none'
    assert list(data.perturbation) == ["none", "CD14", "none", "control", "CHUK", "none"]
    # cell_type: tumor for the tumor full-idx set, else 'other'
    assert list(data.cell_type) == ["other", "tumor", "other", "tumor", "tumor", "other"]
    assert set(data.perturbations()) == {"CD14", "CHUK"} and data.has_ntc is True
    assert data.X.shape == (6, 3) and list(data.gene_names) == genes
    # immune-neighbour annotation carried in meta (a tumor-cell sub-annotation, not a cell type)
    assert data.meta["immune_neighbor_idx"] == [2]
    assert data.meta["immune_distal_idx"] == [4, 5]
