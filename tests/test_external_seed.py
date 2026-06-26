import numpy as np
from spbench.synthetic import make_synthetic
from spbench.external import row_normalize, score_external_seed


def test_row_normalize_per_cell_target_sum():
    X = np.array([[1.0, 1.0, 2.0],      # sum 4
                  [0.0, 0.0, 0.0],      # zero row -> stays zero, no nan
                  [3.0, 0.0, 1.0]])     # sum 4
    out = row_normalize(X, target_sum=1e4)
    nz = np.array([True, False, True])
    assert np.allclose(out[nz].sum(1), 1e4)
    assert np.all(out[1] == 0.0)
    assert not np.isnan(out).any()


def test_row_normalize_default_target():
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(5, 1, size=(7, 4)))
    out = row_normalize(X)
    assert np.allclose(out.sum(1), 1e4)


def test_score_external_seed_returns_metrics():
    data = make_synthetic(0)
    P = data.perturbations()[0]
    pred_full = data.X + 0.5            # mock end-to-end prediction over ALL cells
    r = score_external_seed(data, P, pred_full)
    n_centers = int((data.perturbation == P).sum())
    assert "pcc_delta" in r and "mse" in r and "n" in r
    assert r["n"] == n_centers
    # finite or nan, but never raises
    assert np.isfinite(r["pcc_delta"]) or np.isnan(r["pcc_delta"])
    assert np.isfinite(r["mse"]) or np.isnan(r["mse"])


def test_score_external_seed_missing_perturbation():
    data = make_synthetic(0)
    pred_full = data.X + 0.5
    r = score_external_seed(data, "NOT_A_REAL_PERTURBATION", pred_full)
    assert r["n"] == 0
    assert np.isnan(r["pcc_delta"]) and np.isnan(r["mse"])
