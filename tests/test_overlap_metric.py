# tests/test_overlap_metric.py
import numpy as np
import pytest

from spbench.metrics import get_metric, list_metrics
from spbench.compare import compare_to_baseline
from spbench.annotators import get_annotator


# --- registered Overlap metric (peer of pcc_delta) ------------------------

def test_overlap_metric_registered_active_higher_better():
    m = get_metric("overlap")
    assert m.higher_is_better is True
    assert m.status == "active"
    assert "overlap" in list_metrics(active_only=True)     # a scoring currency like pcc_delta


def test_overlap_metric_identical_is_one_disjoint_is_zero():
    m = get_metric("overlap")
    c = np.array([[0.5, 0.3, 0.2]])
    assert m.compute(c, c.copy()) == pytest.approx(1.0)
    assert m.compute(np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]])) == pytest.approx(0.0)


# --- per-method overlap in the niche board, parallel to pcc ---------------

def _AB_annotator():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 0.1, (20, 2)); A[:, 0] += 5      # type A: high g0
    B = rng.normal(0, 0.1, (20, 2)); B[:, 1] += 5      # type B: high g1
    ann = get_annotator("marker", n_markers=1).fit(
        np.vstack([A, B]), np.array(["A"] * 20 + ["B"] * 20, object), gene_names=["g0", "g1"])
    return ann, A, B


def test_compare_overlap_parallel_to_pcc():
    ann, A, B = _AB_annotator()
    # observed niche = A cells; reference = B cells; method "1" predicts A (perfect), "2" predicts B
    niches = {"observed": A, "reference": B, "1": A, "2": B, "3": A, "4": A}
    out = compare_to_baseline(niches, annotator=ann)
    assert "overlap" in out and "pcc" in out
    assert set(out["overlap"]) == set(out["pcc"])          # same methods scored as pcc
    assert out["overlap"]["GT+base"] == pytest.approx(1.0)  # cloud "1" == observed -> Overlap 1
    assert out["overlap"]["null"] < 0.5                     # reference B vs observed A


def test_compare_overlap_absent_without_annotator():
    niches = {"observed": np.zeros((5, 2)), "reference": np.zeros((5, 2)), "1": np.zeros((5, 2))}
    out = compare_to_baseline(niches)
    assert "overlap" not in out                            # back-compat: opt-in via annotator
    assert "pcc" in out


def test_run_benchmark_reports_overlap_parallel_to_pcc():
    from spbench.config import run_benchmark
    from spbench.synthetic import make_synthetic_with_effects
    data = make_synthetic_with_effects(seed=0)
    ann = get_annotator("marker", n_markers=2).fit(
        data.X, data.cell_type, gene_names=data.gene_names)
    res = run_benchmark(data, perturbations=["SIG"], k=8,
                        gcn_kwargs={"hidden": 16, "epochs": 3}, progress=False, annotator=ann)
    board = res["compare"]["SIG"]
    assert "overlap" in board and "pcc" in board
    assert set(board["overlap"]) == set(board["pcc"])      # Overlap scored for every method pcc is

