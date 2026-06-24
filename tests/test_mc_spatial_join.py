import csv

import pytest

from spbench.mc_spatial_join import (
    classify_quadrant,
    load_dual_metrics,
    join_quadrants,
)


# Mirrors mc_spatial/visualize.py::classify_guide label strings exactly.
_BOTH = "Both (Systemic)"
_XONLY = "X-Only (Self)"
_YONLY = "Y-Only (Niche)"
_INERT = "Inert"

# Real *_Dual_Metrics.csv columns (pipeline.py:504-506 + n_perturbed).
_FIELDS = [
    "guide",
    "n_perturbed",
    "z_score_global",
    "p_val_global",
    "p_adj_global",
    "z_score_specificity",
    "p_val_specificity",
    "p_adj_specificity",
]
# One row per quadrant under raw p<0.05; p_adj_specificity for gX is 0.20 (NOT sig under adj),
# so the p_mode switch is observable (gX: X-Only under raw -> Inert under adj).
_ROWS = [
    {"guide": "gBoth", "n_perturbed": 50, "z_score_global": 3.0, "p_val_global": 0.01,
     "p_adj_global": 0.02, "z_score_specificity": 3.2, "p_val_specificity": 0.01,
     "p_adj_specificity": 0.02},
    {"guide": "gX", "n_perturbed": 40, "z_score_global": 0.5, "p_val_global": 0.20,
     "p_adj_global": 0.25, "z_score_specificity": 4.0, "p_val_specificity": 0.001,
     "p_adj_specificity": 0.20},
    {"guide": "gY", "n_perturbed": 30, "z_score_global": 3.1, "p_val_global": 0.002,
     "p_adj_global": 0.004, "z_score_specificity": 0.8, "p_val_specificity": 0.30,
     "p_adj_specificity": 0.35},
    {"guide": "gInert", "n_perturbed": 20, "z_score_global": 0.2, "p_val_global": 0.60,
     "p_adj_global": 0.70, "z_score_specificity": 0.3, "p_val_specificity": 0.40,
     "p_adj_specificity": 0.45},
]


def _write_dual_csv(path):
    """Write a mock *_Dual_Metrics.csv with the real columns, via stdlib csv (no pandas)."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(_ROWS)


def test_classify_quadrant_four_cells():
    # sig defined as p < p_cutoff (default 0.05), matching classify_guide.
    assert classify_quadrant(0.01, 0.01) == _BOTH
    assert classify_quadrant(0.001, 0.20) == _XONLY
    assert classify_quadrant(0.30, 0.002) == _YONLY
    assert classify_quadrant(0.40, 0.60) == _INERT


def test_classify_quadrant_boundary_is_strict():
    # p exactly at cutoff is NOT significant (strict < like classify_guide).
    assert classify_quadrant(0.05, 0.05) == _INERT
    # custom cutoff honored
    assert classify_quadrant(0.04, 0.60, p_cutoff=0.01) == _INERT


def test_load_dual_metrics_raw(tmp_path):
    csv_path = tmp_path / "Batch_10_Slice_0_Dual_Metrics.csv"
    _write_dual_csv(csv_path)
    out = load_dual_metrics(str(csv_path), p_cutoff=0.05, p_mode="raw")

    # guide copied to perturbation; quadrant added; guide key dropped.
    assert all("perturbation" in r and "quadrant" in r for r in out)
    assert all("guide" not in r for r in out)
    mapping = {r["perturbation"]: r["quadrant"] for r in out}
    assert mapping == {
        "gBoth": _BOTH,
        "gX": _XONLY,
        "gY": _YONLY,
        "gInert": _INERT,
    }
    # numeric metric columns coerced to float for downstream stratified reporting.
    one = out[0]
    assert isinstance(one["z_score_specificity"], float)
    assert isinstance(one["z_score_global"], float)


def test_load_dual_metrics_adj_changes_classification(tmp_path):
    csv_path = tmp_path / "x_Dual_Metrics.csv"
    _write_dual_csv(csv_path)
    out = load_dual_metrics(str(csv_path), p_cutoff=0.05, p_mode="adj")
    mapping = {r["perturbation"]: r["quadrant"] for r in out}
    # gX p_adj_specificity=0.20 (>0.05) -> no longer X-Only, becomes Inert.
    assert mapping["gX"] == _INERT
    # gBoth still significant on both adjusted axes.
    assert mapping["gBoth"] == _BOTH


def test_load_dual_metrics_bad_p_mode(tmp_path):
    csv_path = tmp_path / "x_Dual_Metrics.csv"
    _write_dual_csv(csv_path)
    with pytest.raises(ValueError):
        load_dual_metrics(str(csv_path), p_mode="bogus")


def test_load_dual_metrics_missing_guide_col_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sgrna", "p_val_specificity", "p_val_global"])
        w.writeheader()
        w.writerow({"sgrna": "g0", "p_val_specificity": 0.01, "p_val_global": 0.01})
    with pytest.raises(ValueError):
        load_dual_metrics(str(csv_path))


def test_join_quadrants_attaches_labels(tmp_path):
    csv_path = tmp_path / "x_Dual_Metrics.csv"
    _write_dual_csv(csv_path)
    # benchmark scoring records keyed by 'perturbation' (harness.py / viz.py convention).
    records = [
        {"perturbation": "gBoth", "gain_d2": 0.5},
        {"perturbation": "gY", "gain_d2": 0.4},
        {"perturbation": "gInert", "gain_d2": 0.02},
    ]
    joined = join_quadrants(records, str(csv_path))
    assert len(joined) == 3  # left join: no row multiplication
    m = {r["perturbation"]: r["quadrant"] for r in joined}
    assert m == {"gBoth": _BOTH, "gY": _YONLY, "gInert": _INERT}
    # scoring fields survive the join
    assert all("gain_d2" in r for r in joined)
    # inputs are NOT mutated (join returns new dicts)
    assert records[0] == {"perturbation": "gBoth", "gain_d2": 0.5}


def test_join_quadrants_unknown_for_unmatched(tmp_path):
    csv_path = tmp_path / "x_Dual_Metrics.csv"
    _write_dual_csv(csv_path)
    records = [
        {"perturbation": "gBoth", "gain_d2": 0.5},
        {"perturbation": "gNotInCsv", "gain_d2": 0.1},
    ]
    joined = join_quadrants(records, str(csv_path))
    m = {r["perturbation"]: r["quadrant"] for r in joined}
    assert m["gBoth"] == _BOTH
    assert m["gNotInCsv"] == "Unknown"


def test_inert_is_negative_control_group(tmp_path):
    # The Inert quadrant is the negative control: a model should predict ~no effect there.
    # Helper must let the scorer isolate that group cleanly.
    csv_path = tmp_path / "x_Dual_Metrics.csv"
    _write_dual_csv(csv_path)
    records = [
        {"perturbation": p, "gain_d2": g}
        for p, g in [("gBoth", 0.6), ("gX", 0.05), ("gY", 0.5), ("gInert", 0.01)]
    ]
    joined = join_quadrants(records, str(csv_path))
    inert = [r for r in joined if r["quadrant"] == "Inert"]
    assert [r["perturbation"] for r in inert] == ["gInert"]
    # negative-control expectation: its gain is ~0 (sanity on the mock, not a model assertion)
    assert float(inert[0]["gain_d2"]) < 0.05
