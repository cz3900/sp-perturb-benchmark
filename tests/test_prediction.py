# tests/test_prediction.py
import numpy as np
import pytest
from spbench.prediction import StandardPrediction


def test_all_dims_default_none():
    p = StandardPrediction()
    assert p.d1 is None
    assert p.d2 is None
    assert p.d3 is None


def test_holds_arrays_unmodified():
    d1 = np.arange(20, dtype=float)
    d2 = np.ones(20, dtype=float)
    d3 = np.array([0.2, 0.3, 0.5])
    p = StandardPrediction(d1=d1, d2=d2, d3=d3)
    assert np.array_equal(p.d1, d1)
    assert np.array_equal(p.d2, d2)
    assert np.array_equal(p.d3, d3)


def test_partial_fill_d1_only():
    p = StandardPrediction(d1=np.zeros(5))
    assert p.d1 is not None and p.d1.shape == (5,)
    assert p.d2 is None
    assert p.d3 is None


def test_covered_dims_reports_filled_keys():
    assert StandardPrediction(d1=np.zeros(3)).covered_dims() == ("D1",)
    assert StandardPrediction(d1=np.zeros(3), d2=np.zeros(3)).covered_dims() == ("D1", "D2")
    assert StandardPrediction().covered_dims() == ()
    full = StandardPrediction(d1=np.zeros(3), d2=np.zeros(3), d3=np.zeros(2))
    assert full.covered_dims() == ("D1", "D2", "D3")


def test_as_dict_only_covered():
    p = StandardPrediction(d1=np.zeros(3), d3=np.zeros(2))
    d = p.as_dict()
    assert set(d) == {"D1", "D3"}
    assert d["D1"].shape == (3,)
    assert d["D3"].shape == (2,)
