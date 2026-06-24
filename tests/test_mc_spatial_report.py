import math

import pytest

from spbench.mc_spatial_report import (
    DIMENSION_QUADRANTS,
    REPORT_FIELDS,
    stratified_report,
)

_BOTH = "Both (Systemic)"
_XONLY = "X-Only (Self)"
_YONLY = "Y-Only (Niche)"
_INERT = "Inert"


def _joined():
    # One record per perturbation, already carrying a `quadrant` label (as join_quadrants outputs)
    # and per-dimension gain fields (gain>0 beats the no-effect baseline).
    data = [
        # (perturbation, quadrant, gain_d1, gain_d2)
        ("pB", _BOTH, 0.50, 0.60),
        ("pX1", _XONLY, 0.40, 0.05),
        ("pX2", _XONLY, 0.30, -0.10),
        ("pY", _YONLY, 0.10, 0.45),
        ("pI1", _INERT, 0.01, 0.00),
        ("pI2", _INERT, -0.02, 0.02),
    ]
    return [
        {"perturbation": p, "quadrant": q, "gain_d1": g1, "gain_d2": g2}
        for p, q, g1, g2 in data
    ]


def _row(rep, dim, is_nc):
    return [r for r in rep if r["dimension"] == dim and r["is_negative_control"] == is_nc][0]


def test_dimension_quadrants_mapping():
    assert DIMENSION_QUADRANTS["d1"] == {_XONLY, _BOTH}
    assert DIMENSION_QUADRANTS["d2"] == {_YONLY, _BOTH}


def test_report_d1_only_counts_x_signal_quadrants():
    rep = stratified_report(_joined())
    # D1 signal group = Both + X-Only = pB, pX1, pX2  (NOT pY, NOT Inert)
    row = _row(rep, "d1", False)
    assert row["n"] == 3
    assert math.isclose(row["mean_gain"], (0.50 + 0.40 + 0.30) / 3)
    assert math.isclose(row["frac_beat"], 1.0)  # all 3 gains > 0


def test_report_d2_only_counts_y_signal_quadrants():
    rep = stratified_report(_joined())
    # D2 signal group = Both + Y-Only = pB, pY
    row = _row(rep, "d2", False)
    assert row["n"] == 2
    assert math.isclose(row["mean_gain"], (0.60 + 0.45) / 2)
    assert math.isclose(row["frac_beat"], 1.0)


def test_report_inert_is_negative_control_per_dimension():
    rep = stratified_report(_joined())
    nc = [r for r in rep if r["is_negative_control"]]
    # one negative-control row per dimension
    assert {r["dimension"] for r in nc} == {"d1", "d2"}
    assert all(r["quadrant_group"] == "Inert" for r in nc)
    # Inert has 2 perturbations for each dimension
    assert all(r["n"] == 2 for r in nc)
    # negative control: mean gain ~ 0 (model should predict no effect here)
    d1_nc = _row(rep, "d1", True)
    assert math.isclose(d1_nc["mean_gain"], (0.01 + (-0.02)) / 2)
    assert abs(d1_nc["mean_gain"]) < 0.05


def test_report_fields_and_long_shape():
    rep = stratified_report(_joined())
    # every row has exactly REPORT_FIELDS, in order
    assert all(tuple(r.keys()) == REPORT_FIELDS for r in rep)
    # 2 dims x (1 signal group + 1 negative control) = 4 rows
    assert len(rep) == 4


def test_report_empty_group_yields_nan_not_crash():
    # No X-signal perturbations at all -> D1 signal group is empty.
    recs = [
        {"perturbation": "pY", "quadrant": _YONLY, "gain_d1": 0.1, "gain_d2": 0.4},
        {"perturbation": "pI", "quadrant": _INERT, "gain_d1": 0.0, "gain_d2": 0.0},
    ]
    rep = stratified_report(recs)
    d1_sig = _row(rep, "d1", False)
    assert d1_sig["n"] == 0
    assert math.isnan(d1_sig["mean_gain"])
    assert math.isnan(d1_sig["frac_beat"])


def test_report_missing_gain_field_raises():
    recs = [{"perturbation": "pB", "quadrant": _BOTH, "gain_d1": 0.5}]  # no gain_d2
    with pytest.raises(ValueError):
        stratified_report(recs)
