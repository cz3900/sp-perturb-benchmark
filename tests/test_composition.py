# tests/test_composition.py
import numpy as np
import pytest

from spbench.composition import (
    niche_composition, composition_overlap, composition_board, composition_eval,
)
from spbench.synthetic import make_synthetic_with_effects
from spbench.graph import build_knn_graph
from spbench.propagation_gt import propagation_gt
from spbench.annotators import get_annotator


# --- composition vector ---------------------------------------------------

def test_niche_composition_is_simplex():
    cats = np.array(["A", "B", "C"], object)
    c = niche_composition(np.array(["A", "A", "B"], object), cats)
    assert c == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert c.sum() == pytest.approx(1.0)


def test_niche_composition_empty_is_zeros():
    c = niche_composition(np.array([], object), np.array(["A", "B"], object))
    assert np.all(c == 0.0)


# --- Overlap (1 - TV) -----------------------------------------------------

def test_overlap_identical_is_one():
    c = np.array([0.5, 0.3, 0.2])
    assert composition_overlap(c, c) == pytest.approx(1.0)


def test_overlap_disjoint_is_zero():
    assert composition_overlap(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(0.0)


def test_overlap_half_is_intuitive_fraction():
    # predict everything type A; truth is half A half B -> 50% of mass mismatched -> Overlap 0.5
    assert composition_overlap(np.array([1.0, 0.0]), np.array([0.5, 0.5])) == pytest.approx(0.5)


# --- board (pure label space) ---------------------------------------------

def test_board_null_gain_zero_and_perfect_prediction():
    cats = np.array(["A", "B"], object)
    gt = np.array(["A", "A", "B"], object)    # comp [2/3, 1/3]
    ref = np.array(["A", "A", "A"], object)   # comp [1, 0]  -> Overlap(ref,gt)=1-TV; TV=1/3
    board = composition_board(gt, ref, {"perfect": gt.copy()}, cats=cats)
    assert board["overlap"]["null"] == pytest.approx(2 / 3)
    assert board["gain"]["null"] == pytest.approx(0.0)            # baseline gains nothing over itself
    assert board["overlap"]["perfect"] == pytest.approx(1.0)
    assert board["gain"]["perfect"] == pytest.approx(1 / 3)       # closes the whole gap


# --- eval bridge: observed D3 shift via native labels ---------------------

def test_eval_detects_d3_shift_for_sig_not_inert():
    data = make_synthetic_with_effects(seed=0)
    edges = build_knn_graph(data, k=8)
    sig = composition_eval(data, "SIG", edges)        # annotator=None -> native cell_type
    inert = composition_eval(data, "INERT", edges)
    # SIG plants a niche cell-type enrichment -> predicting 'no change' overlaps the truth LESS
    assert sig["overlap"]["null"] < inert["overlap"]["null"]
    assert inert["overlap"]["null"] > 0.85            # inert niche composition ~ unchanged


# --- eval bridge: frozen annotator on observed + predicted expression -----

def test_eval_with_annotator_oracle_prediction_is_perfect():
    data = make_synthetic_with_effects(seed=0)
    edges = build_knn_graph(data, k=8)
    ann = get_annotator("marker", n_markers=3).fit(
        data.X, data.cell_type, gene_names=data.gene_names)
    g = propagation_gt(data, "SIG", edges)
    # an oracle whose predicted niche expression == the observed niche expression must,
    # through the SAME frozen annotator, reproduce the gt composition exactly -> Overlap 1.
    board = composition_eval(data, "SIG", edges, annotator=ann,
                             pred_niches={"oracle": data.X[g["pert_nb"]]})
    assert board["overlap"]["oracle"] == pytest.approx(1.0)


# --- pipeline wiring ------------------------------------------------------

def test_run_benchmark_composition_off_by_default():
    from spbench.config import run_benchmark
    data = make_synthetic_with_effects(seed=0)
    res = run_benchmark(data, perturbations=["SIG"], k=8, k_ref=5,
                        gcn_kwargs={"hidden": 16, "epochs": 3}, progress=False)
    assert "composition" not in res            # back-compat: opt-in only


def test_run_benchmark_scores_composition_board():
    from spbench.config import run_benchmark
    data = make_synthetic_with_effects(seed=0)
    res = run_benchmark(data, perturbations=["SIG", "INERT"], k=8, k_ref=5,
                        gcn_kwargs={"hidden": 16, "epochs": 3}, progress=False,
                        composition=True)
    assert "composition" in res and "SIG" in res["composition"]
    # the planted D3 enrichment makes 'predict no change' overlap the SIG truth less than INERT's
    assert (res["composition"]["SIG"]["overlap"]["null"]
            < res["composition"]["INERT"]["overlap"]["null"])
