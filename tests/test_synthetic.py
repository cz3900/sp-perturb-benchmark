import numpy as np
from spbench.synthetic import make_synthetic

def test_shapes_and_labels():
    d = make_synthetic(seed=0)
    assert d.n_cells > 0 and d.n_genes == 20
    assert d.is_control.sum() > 0
    assert len(d.perturbations()) == 3   # P0, P1, P2

def test_planted_seed_shift_is_recoverable():
    d = make_synthetic(seed=0)
    g = d.meta["planted"]["P0"]["seed_gene"]
    pert_mean = d.X[d.perturbation == "P0", g].mean()
    ctrl_mean = d.X[d.is_control, g].mean()
    assert pert_mean > ctrl_mean + 0.5

def test_planted_propagation_shift_on_neighbors():
    d = make_synthetic(seed=0)
    assert d.meta["planted"]["P0"]["prop_gene"] != d.meta["planted"]["P0"]["seed_gene"]
