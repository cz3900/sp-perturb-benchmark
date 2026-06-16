import numpy as np
from spbench.metrics import get_metric

def test_perfect_profile_correlation_is_one():
    m = get_metric("rho_niche")
    pred = {"nicheA": np.array([1.0, -2.0]), "nicheB": np.array([0.5, 0.0])}
    true = {"nicheA": np.array([1.0, -2.0]), "nicheB": np.array([0.5, 0.0])}
    assert abs(m.compute(pred, true) - 1.0) < 1e-9

def test_anticorrelated_profile_is_negative():
    m = get_metric("rho_niche")
    pred = {"a": np.array([1.0, 2.0]), "b": np.array([-1.0, -2.0])}
    true = {"a": np.array([-1.0, -2.0]), "b": np.array([1.0, 2.0])}
    assert m.compute(pred, true) < 0
