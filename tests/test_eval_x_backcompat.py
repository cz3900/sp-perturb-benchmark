# tests/test_eval_x_backcompat.py
import numpy as np
import pytest
from spbench.compare import evaluate_seed, compare_to_baseline


def _seed_niches():
    rng = np.random.default_rng(7)
    G = 6
    ref = rng.random((12, G)) * 3.0
    pred = ref.mean(0) + 1.0 + rng.random((8, G)) * 0.1
    obs = ref.mean(0) + 1.0 + rng.random((9, G)) * 0.1
    return {"seed_obs": obs, "seed_pred": pred, "seed_ref": ref}


def _niche_niches():
    rng = np.random.default_rng(8)
    G = 5
    ref = rng.random((30, G)) * 4.0
    obs = ref.mean(0) + 2.0 + rng.random((30, G)) * 0.2
    c = ref.mean(0) + 1.5 + rng.random((30, G)) * 0.2
    return {"observed": obs, "reference": ref, "1": c, "2": c.copy(),
            "3": c.copy(), "4": c.copy()}


def _assert_scalar_equal(a, b):
    """Equal as scalars, treating NaN == NaN as equal."""
    if np.isnan(a) or np.isnan(b):
        assert np.isnan(a) and np.isnan(b)
    else:
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12)


def test_evaluate_seed_none_equals_legacy_call():
    n = _seed_niches()
    # explicit eval_X=None must equal the no-arg legacy invocation, field by field
    a = evaluate_seed(n)
    b = evaluate_seed(n, eval_X=None)
    _assert_scalar_equal(a["pcc_delta"], b["pcc_delta"])
    _assert_scalar_equal(a["mse"], b["mse"])
    assert a["n"] == b["n"]


def test_compare_none_equals_legacy_call():
    n = _niche_niches()
    a = compare_to_baseline(n)
    b = compare_to_baseline(n, eval_X=None)
    for k in a["pcc"]:
        _assert_scalar_equal(a["pcc"][k], b["pcc"][k])
        _assert_scalar_equal(a["mag"][k], b["mag"][k])
    assert a["n"] == b["n"]
