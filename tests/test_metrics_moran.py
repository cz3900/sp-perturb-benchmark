import numpy as np
from spbench.metrics.moran import morans_i

def test_clustered_field_positive():
    g = 10
    xs, ys = np.meshgrid(np.arange(g), np.arange(g))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    vals = (coords[:, 0] < g / 2).astype(float)
    assert morans_i(vals, coords, k=4) > 0.5

def test_random_field_near_zero():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, size=(200, 2))
    vals = rng.normal(size=200)
    assert abs(morans_i(vals, coords, k=6)) < 0.2
