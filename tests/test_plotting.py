import numpy as np
from spbench.config import run_benchmark
from spbench.plotting import collect_prop_samples, collect_seed_samples

GCN_KW = {"hidden": 16, "epochs": 5}


def test_collect_prop_samples_has_named_gcn(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    boxes, dashed = collect_prop_samples(res)
    assert "GCN" in boxes, "learned prop must show up named 'GCN'"
    assert "Gaussian" in boxes
    assert isinstance(boxes["GCN"], np.ndarray) and boxes["GCN"].size > 0
    assert "null" in dashed and "oracle" in dashed


def test_collect_seed_samples(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    boxes, dashed = collect_seed_samples(res)
    assert len(boxes) >= 1
    assert "null" in dashed
