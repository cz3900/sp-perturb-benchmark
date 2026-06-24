# tests/test_metrics_comp.py
import numpy as np
import pytest
from spbench.metrics import get_metric, list_metrics


def test_comp_l1_registered():
    assert "comp_l1" in list_metrics()
    m = get_metric("comp_l1")
    assert m.higher_is_better is False


def test_comp_l1_identical_compositions_is_zero():
    m = get_metric("comp_l1")
    p = np.array([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]], float)
    assert m.compute(p, p.copy()) == pytest.approx(0.0, abs=1e-12)


def test_comp_l1_is_total_variation_of_group_means():
    m = get_metric("comp_l1")
    pred = np.array([[1.0, 0.0, 0.0]], float)   # mean comp = [1,0,0]
    obs = np.array([[0.0, 1.0, 0.0]], float)    # mean comp = [0,1,0]
    # TV = 0.5 * (|1-0| + |0-1| + |0-0|) = 1.0
    assert m.compute(pred, obs) == pytest.approx(1.0, abs=1e-12)


def test_comp_l1_handles_zeros_without_nan():
    m = get_metric("comp_l1")
    pred = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], float)
    obs = np.array([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]], float)
    out = m.compute(pred, obs)
    # mean pred = [0,0,1], mean obs = [0,0.5,0.5]; TV = 0.5*(0+0.5+0.5)=0.5
    assert np.isfinite(out)
    assert out == pytest.approx(0.5, abs=1e-12)


def test_comp_l1_empty_is_nan():
    m = get_metric("comp_l1")
    assert np.isnan(m.compute(np.zeros((0, 3)), np.zeros((0, 3))))
