import numpy as np
from spbench.metrics import get_metric

def test_identical_distributions_zero():
    e = get_metric("energy")
    X = np.array([[1.0], [3.0]])
    assert abs(e.compute(X, X.copy())) < 1e-9

def test_known_value():
    # X={1,3}, Y={2,4}: 2*1.5 - 1 - 1 = 1
    e = get_metric("energy")
    X = np.array([[1.0], [3.0]]); Y = np.array([[2.0], [4.0]])
    assert abs(e.compute(X, Y) - 1.0) < 1e-9

def test_registry_lists_energy():
    from spbench.metrics import list_metrics
    assert "energy" in list_metrics()
