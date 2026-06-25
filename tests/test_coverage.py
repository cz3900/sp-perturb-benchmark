import pytest
from spbench.coverage import guide_overlap

def test_guide_overlap_splits_in_and_out():
    out = guide_overlap(["A", "B", "C"], ["B", "C", "D"])
    assert out["in"] == ["B", "C"]
    assert out["out"] == ["A"]

def test_guide_overlap_empty_allowed_all_out():
    out = guide_overlap(["A", "B"], [])
    assert out["in"] == [] and out["out"] == ["A", "B"]
