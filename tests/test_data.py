import numpy as np
from spbench.data import StandardData

def _toy():
    return StandardData(
        X=np.arange(12, dtype=float).reshape(4, 3),
        coords=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float),
        perturbation=np.array(["TP53", "control", "none", "TP53"]),
        cell_type=np.array(["Hep", "Hep", "Endo", "Hep"]),
        batch=np.array(["s0", "s0", "s0", "s0"]),
        gene_names=["g1", "g2", "g3"],
        meta={"name": "toy"},
    )

def test_flags():
    d = _toy()
    assert d.n_cells == 4 and d.n_genes == 3
    assert d.is_control.tolist() == [False, True, False, False]
    assert d.is_perturbed.tolist() == [True, False, False, True]
    assert d.is_unlabeled.tolist() == [False, False, True, False]

def test_perturbations_excludes_control_and_none():
    assert _toy().perturbations() == ["TP53"]

def test_subset_keeps_fields():
    d = _toy().subset(np.array([0, 1]))
    assert d.n_cells == 2 and d.gene_names == ["g1", "g2", "g3"]
    assert d.perturbation.tolist() == ["TP53", "control"]
