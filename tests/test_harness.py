import numpy as np
from spbench.graph import build_knn_graph
from spbench.models.trivial_seed import TrivialSeed
from spbench.models.gaussian_prop import GaussianProp
from spbench.models.gcn_prop import SimpleGCN
from spbench.harness import fill_2x2

def test_grid_has_four_cells_with_scores(synth):
    edges = build_knn_graph(synth, k=8)
    seed = TrivialSeed().fit(synth)
    base = GaussianProp().fit(synth, edges)
    learned = SimpleGCN(hidden=16, epochs=3).fit(synth, edges)
    grid = fill_2x2(synth, "P0", edges, seed, base, learned)
    for cell in ["1", "2", "3", "4"]:
        assert "pcc_prop" in grid[cell]
        # niche PCC-delta is bounded [-1, 1] (or NaN on a degenerate flat shift)
        v = grid[cell]["pcc_prop"]
        assert np.isnan(v) or -1.0 <= v <= 1.0

def test_baseline_column_does_not_leak(synth):
    # Regression: with GT seed, the Gaussian baseline must NOT reproduce the observed niche
    # (which would be PCC-delta ~1, i.e. propagation started from the observed matrix). The
    # harness must start propagation from a CONTROL reference state, so cell (1) is non-trivial.
    from spbench.config import run_benchmark
    res = run_benchmark(synth, perturbations=["P0"], k=8, gcn_kwargs={"hidden": 16, "epochs": 3})
    g = res["grids"]["P0"]
    assert g["1"]["pcc_prop"] < 1.0 - 1e-3, "baseline+GT-seed reproduced observed niche (leak)"
    assert res["leakage_pass"]["P0"]
