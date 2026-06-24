import numpy as np
import pytest
from spbench.synthetic import make_synthetic_with_effects
from spbench.data import StandardData, CONTROL, UNLABELED


def test_returns_standard_data_with_effects_meta():
    d = make_synthetic_with_effects(seed=0)
    assert isinstance(d, StandardData)
    assert d.n_cells > 0
    assert "effects" in d.meta
    eff = d.meta["effects"]
    # both a significant perturbation and an inert (negative-control) perturbation exist
    assert "significant" in eff and "inert" in eff
    assert len(eff["significant"]) >= 1
    assert len(eff["inert"]) >= 1
    # significant and inert label sets are disjoint and both appear as real perturbations
    sig = set(eff["significant"]); inert = set(eff["inert"])
    assert sig.isdisjoint(inert)
    present = set(d.perturbations())
    assert sig <= present and inert <= present


def test_significant_perturbation_carries_all_three_dims():
    d = make_synthetic_with_effects(seed=0)
    eff = d.meta["effects"]
    p = eff["significant"][0]
    spec = eff["spec"][p]
    # D1: a seed gene with a known positive shift on the perturbed cells themselves
    assert "d1_gene" in spec and "d1_shift" in spec and spec["d1_shift"] > 0
    # D2: a (distinct) propagation gene with a known shift on bystander neighbors
    assert "d2_gene" in spec and "d2_shift" in spec and spec["d2_shift"] > 0
    assert spec["d1_gene"] != spec["d2_gene"]
    # D3: a known cell-type whose niche composition is enriched near perturbed centers
    assert "d3_cell_type" in spec and "d3_extra" in spec and spec["d3_extra"] > 0


def test_inert_perturbation_has_no_injected_effect():
    d = make_synthetic_with_effects(seed=0)
    eff = d.meta["effects"]
    p = eff["inert"][0]
    spec = eff["spec"][p]
    assert spec["d1_shift"] == 0.0
    assert spec["d2_shift"] == 0.0
    assert spec["d3_extra"] == 0.0


def test_d1_seed_shift_recoverable_for_significant():
    d = make_synthetic_with_effects(seed=0)
    eff = d.meta["effects"]
    p = eff["significant"][0]
    g = eff["spec"][p]["d1_gene"]
    shift = eff["spec"][p]["d1_shift"]
    pert_mean = d.X[d.perturbation == p, g].mean()
    ctrl_mean = d.X[d.is_control, g].mean()
    assert pert_mean > ctrl_mean + 0.5 * shift


def test_d1_seed_no_shift_for_inert():
    d = make_synthetic_with_effects(seed=0)
    eff = d.meta["effects"]
    p = eff["inert"][0]
    g = eff["spec"][p]["d1_gene"]      # the gene that *would* have been the seed gene
    pert_mean = d.X[d.perturbation == p, g].mean()
    ctrl_mean = d.X[d.is_control, g].mean()
    assert abs(pert_mean - ctrl_mean) < 0.5   # essentially no shift


def test_d2_propagation_shift_on_neighbors_for_significant():
    from spbench.graph import build_knn_graph, neighbors_of
    d = make_synthetic_with_effects(seed=0, grid=24)
    eff = d.meta["effects"]
    p = eff["significant"][0]
    pg = eff["spec"][p]["d2_gene"]
    edges = build_knn_graph(d, k=8)
    centers = np.where(d.perturbation == p)[0]
    nb = np.unique(np.concatenate(
        [neighbors_of(c, edges) for c in centers]))
    nb = nb[~d.is_perturbed[nb]]                 # bystanders only
    ctrl = d.is_control
    assert d.X[nb, pg].mean() > d.X[ctrl, pg].mean() + 0.3


def test_d3_composition_enriched_for_significant():
    from spbench.graph import build_knn_graph, neighbors_of
    d = make_synthetic_with_effects(seed=0, grid=24)
    eff = d.meta["effects"]
    p = eff["significant"][0]
    ct = eff["spec"][p]["d3_cell_type"]
    edges = build_knn_graph(d, k=8)
    centers = np.where(d.perturbation == p)[0]
    nb = np.concatenate([neighbors_of(c, edges) for c in centers])
    nb = nb[~d.is_perturbed[nb]]
    near_frac = (d.cell_type[nb] == ct).mean()
    bg_frac = (d.cell_type == ct).mean()
    assert near_frac > bg_frac + 0.05            # enriched vs background


def test_determinism():
    a = make_synthetic_with_effects(seed=3)
    b = make_synthetic_with_effects(seed=3)
    assert np.allclose(a.X, b.X)
    assert (a.perturbation == b.perturbation).all()
    assert (a.cell_type == b.cell_type).all()
