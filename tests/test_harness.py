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
        assert "energy_prop" in grid[cell]
        assert np.isfinite(grid[cell]["energy_prop"])

def test_baseline_column_does_not_leak(synth):
    # Regression: with GT seed, the Gaussian baseline must NOT reproduce the observed niche
    # (which happens if propagation starts from the observed matrix -> energy ~0). The harness
    # must start propagation from a CONTROL reference state, so cell (1) is clearly non-trivial.
    from spbench.config import run_benchmark
    res = run_benchmark(synth, perturbations=["P0"], k=8, gcn_kwargs={"hidden": 16, "epochs": 3})
    g = res["grids"]["P0"]
    assert g["1"]["energy_prop"] > 0.05, "baseline+GT-seed reproduced observed niche (leak)"
    assert res["leakage_pass"]["P0"]
