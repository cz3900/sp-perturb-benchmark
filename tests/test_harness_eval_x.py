# tests/test_harness_eval_x.py
import numpy as np
import pytest
from spbench.graph import build_knn_graph
from spbench.models.trivial_seed import TrivialSeed
from spbench.models.gaussian_prop import GaussianProp
from spbench.models.gcn_prop import SimpleGCN
from spbench.harness import fill_2x2
from spbench.compare import evaluate_seed, compare_to_baseline


def _models(synth, edges):
    seed = TrivialSeed().fit(synth)
    base = GaussianProp().fit(synth, edges)
    learned = SimpleGCN(hidden=16, epochs=3).fit(synth, edges)
    return seed, base, learned


def test_fill_2x2_default_no_eval_x_key(synth):
    edges = build_knn_graph(synth, k=8)
    seed, base, learned = _models(synth, edges)
    g = fill_2x2(synth, "P0", edges, seed, base, learned, return_niches=True)
    # default: eval_X carried as None
    assert g["_niches"]["eval_X"] is None
    for cell in ["1", "2", "3", "4"]:
        assert np.isfinite(g[cell]["energy_prop"])


def test_fill_2x2_stashes_eval_x(synth):
    edges = build_knn_graph(synth, k=8)
    seed, base, learned = _models(synth, edges)
    g = fill_2x2(synth, "P0", edges, seed, base, learned,
                 return_niches=True, eval_X=np.arcsinh)
    assert g["_niches"]["eval_X"] is np.arcsinh
    # energy grid untouched by eval_X (still raw space, finite)
    for cell in ["1", "2", "3", "4"]:
        assert np.isfinite(g[cell]["energy_prop"])


def test_eval_x_flows_to_downstream_scorers(synth):
    edges = build_knn_graph(synth, k=8)
    seed, base, learned = _models(synth, edges)
    g = fill_2x2(synth, "P0", edges, seed, base, learned,
                 return_niches=True, eval_X=np.arcsinh)
    niches = g["_niches"]
    ex = niches["eval_X"]
    se = evaluate_seed(niches, eval_X=ex)
    cb = compare_to_baseline(niches, repeats=3, seed=0, eval_X=ex)
    # arcsinh is finite on the signed real niches (log1p would NaN here)
    assert np.isfinite(se["pcc_delta"]) and np.isfinite(se["mse"])
    assert np.isfinite(cb["pcc"]["model+learned"])
