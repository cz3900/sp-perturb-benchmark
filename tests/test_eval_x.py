# tests/test_eval_x.py
import numpy as np
import pytest
from spbench.compare import evaluate_seed, _apply_eval_X


def _niches():
    rng = np.random.default_rng(0)
    G = 6
    ref = rng.random((10, G)) * 3.0
    pred = ref.mean(0) + 1.5 + rng.random((8, G)) * 0.1   # clear positive shift over ref
    obs = ref.mean(0) + 1.5 + rng.random((9, G)) * 0.1
    return {"seed_obs": obs, "seed_pred": pred, "seed_ref": ref}


def test_apply_eval_x_none_is_identity():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = _apply_eval_X(A, None)
    assert out is A  # identity short-circuit, no copy


def test_apply_eval_x_callable_is_applied():
    A = np.array([[-2.0, 0.0, 3.0]])
    out = _apply_eval_X(A, np.arcsinh)
    np.testing.assert_allclose(out, np.arcsinh(A))
    assert np.isfinite(out).all()   # finite even on the negative entry


def test_evaluate_seed_default_unchanged():
    n = _niches()
    out = evaluate_seed(n)
    assert set(out) == {"pcc_delta", "mse", "mag", "n"}
    assert out["n"] == 9
    assert np.isfinite(out["pcc_delta"])
    assert np.isfinite(out["mse"])


def test_evaluate_seed_eval_x_changes_scores_and_stays_finite():
    n = _niches()
    base = evaluate_seed(n)
    logged = evaluate_seed(n, eval_X=np.arcsinh)
    assert np.isfinite(logged["pcc_delta"])
    assert np.isfinite(logged["mse"])
    # arcsinh compresses magnitude -> MSE strictly smaller than in raw space here
    assert logged["mse"] < base["mse"]
    assert logged["n"] == base["n"]


def test_evaluate_seed_empty_with_eval_x():
    out = evaluate_seed({"seed_obs": np.zeros((0, 4))}, eval_X=np.arcsinh)
    assert out["n"] == 0
    assert np.isnan(out["pcc_delta"]) and np.isnan(out["mse"])
