import numpy as np
import pytest
from spbench.synthetic import make_synthetic_with_effects
from spbench.graph import build_knn_graph, neighbors_of
from spbench.niche import compute_niche_composition
from spbench.propagation_gt import propagation_gt
from spbench.compare import compare_to_baseline
from spbench.harness import _control_residuals
from spbench.metrics import get_metric
from spbench.metrics.pcc_delta import delta_corr


def _bystanders(d, centers, edges):
    nb = np.concatenate([neighbors_of(c, edges) for c in centers])
    return nb[~d.is_perturbed[nb]]


def test_d1_recovery_significant_vs_inert():
    """D1 (seed): the injected significant perturbation's gene-wise shift must correlate
    with the planted seed direction (delta_corr > 0.5); the inert one's mean profile must
    match control (MSE small)."""
    d = make_synthetic_with_effects(seed=0)
    eff = d.meta["effects"]
    ctrl = d.X[d.is_control]

    for p in eff["significant"]:
        obs = d.X[d.perturbation == p]
        g = eff["spec"][p]["d1_gene"]
        true = np.zeros(d.n_genes); true[g] = eff["spec"][p]["d1_shift"]   # only seed gene moved
        pred_delta = obs.mean(0) - ctrl.mean(0)
        assert delta_corr(pred_delta, true) > 0.5                          # recovered direction

    # inert: observed perturbed cells == the control mean profile (no seed magnitude shift)
    p = eff["inert"][0]
    obs = d.X[d.perturbation == p]
    assert get_metric("mse").compute(obs, ctrl) < 0.5


def test_d2_propagation_recovered_for_significant_zero_for_inert():
    """D2: the propagation gene's niche mean exceeds the reference niche mean for each
    significant perturbation, and matches it for the inert one."""
    d = make_synthetic_with_effects(seed=0, grid=24)
    eff = d.meta["effects"]
    edges = build_knn_graph(d, k=8)

    for p in eff["significant"]:
        gt = propagation_gt(d, p, edges, k_ref=5)
        obs, ref = gt["perturbed_niche"], gt["reference_niche"]
        pg = eff["spec"][p]["d2_gene"]
        assert obs[:, pg].mean() > ref[:, pg].mean() + 0.2

    p = eff["inert"][0]
    gt = propagation_gt(d, p, edges, k_ref=5)
    obs, ref = gt["perturbed_niche"], gt["reference_niche"]
    pg = eff["spec"][p]["d2_gene"]
    assert abs(obs[:, pg].mean() - ref[:, pg].mean()) < 0.3


def test_compare_has_effect_flag_tracks_injection():
    """The compare orchestration's has_effect flag (with the oracle floor supplied via
    residuals) must be True for the significant perturbation and False for the inert one,
    and the inert null distance must be strictly smaller than the significant one's."""
    d = make_synthetic_with_effects(seed=0, grid=24)
    eff = d.meta["effects"]
    edges = build_knn_graph(d, k=8)
    resid = _control_residuals(d)          # oracle floor; without it has_effect is always True

    def niches(p):
        gt = propagation_gt(d, p, edges, k_ref=5)
        return {"observed": gt["perturbed_niche"], "reference": gt["reference_niche"]}

    sig = eff["significant"][0]
    res_sig = compare_to_baseline(niches(sig), residuals=resid, repeats=10, seed=0)
    assert res_sig["has_effect"] is True

    inert = eff["inert"][0]
    res_inert = compare_to_baseline(niches(inert), residuals=resid, repeats=10, seed=0)
    assert res_inert["e"]["null"] < res_sig["e"]["null"]
    assert res_inert["has_effect"] is False


def test_uninjected_dimension_is_near_zero():
    """A gene that was NOT injected for the significant perturbation shows ~no shift in
    either the seed cells (D1) or the niche (D2)."""
    d = make_synthetic_with_effects(seed=0, grid=24)
    eff = d.meta["effects"]
    p = eff["significant"][0]
    used = {eff["spec"][p]["d1_gene"], eff["spec"][p]["d2_gene"]}
    quiet = next(g for g in range(d.n_genes) if g not in used)

    ctrl = d.X[d.is_control]
    obs = d.X[d.perturbation == p]
    assert abs(obs[:, quiet].mean() - ctrl[:, quiet].mean()) < 0.5

    edges = build_knn_graph(d, k=8)
    centers = np.where(d.perturbation == p)[0]
    nb = _bystanders(d, centers, edges)
    assert abs(d.X[nb, quiet].mean() - ctrl[:, quiet].mean()) < 0.5


def test_d3_composition_recovered_for_significant_zero_for_inert():
    """D3 (niche composition): the pipeline's compute_niche_composition path must RECOVER the
    injected over-enrichment of `d3_cell_type` among the neighbours of each significant
    perturbation's centers, and judge the inert perturbation effect-free.

    We build the graph, compute the per-cell niche cell-type composition, and score the
    comp_l1 (TV) distance between the perturbed centers' niche compositions and the control
    reference niche. It must be LARGE (> 0.15) for each significant perturbation and ~0
    (< 0.10) for the inert one, with each significant distance strictly above the inert one.
    Sanity: this would fail if D3 recovery were broken — with no composition enrichment
    injected (d3_extra=0) the significant distances collapse to <0.06 (verified)."""
    d = make_synthetic_with_effects(seed=0, grid=24)
    eff = d.meta["effects"]
    edges = build_knn_graph(d, k=8)
    comp = compute_niche_composition(d, edges)          # (n_cells, C) row-simplex niches
    comp_l1 = get_metric("comp_l1")
    ref_comp = comp[d.is_control]                        # control/reference niche

    inert = eff["inert"][0]
    inert_centers = np.where(d.perturbation == inert)[0]
    inert_dist = comp_l1.compute(comp[inert_centers], ref_comp)
    assert inert_dist < 0.10                             # inert: no composition effect

    for p in eff["significant"]:
        centers = np.where(d.perturbation == p)[0]
        sig_dist = comp_l1.compute(comp[centers], ref_comp)
        assert sig_dist > 0.15                           # significant: D3 recovered
        assert sig_dist > inert_dist                     # and strictly above the inert floor

        # the recovered enrichment is on the *injected* cell type, not some other one
        cats = sorted(set(d.cell_type))
        ci = cats.index(eff["spec"][p]["d3_cell_type"])
        assert comp[centers].mean(0)[ci] > ref_comp.mean(0)[ci] + 0.1
