import numpy as np
from scipy.spatial.distance import cdist
from spbench.graph import build_knn_graph
from spbench.models.trivial_seed import TrivialSeed
from spbench.models.gaussian_prop import GaussianProp
from spbench.models.gcn_prop import SimpleGCN
from spbench.harness import fill_2x2, _control_residuals

def test_grid_has_four_cells_with_scores(synth):
    edges = build_knn_graph(synth, k=8)
    seed = TrivialSeed().fit(synth)
    base = GaussianProp().fit(synth, edges)
    learned = SimpleGCN(hidden=16, epochs=3).fit(synth, edges)
    grid = fill_2x2(synth, "P0", edges, seed, base, learned)
    for cell in ["1", "2", "3", "4"]:
        assert "energy_prop" in grid[cell]
        assert np.isfinite(grid[cell]["energy_prop"])

def _niche_spread(A):
    A = np.asarray(A, float)
    return float(cdist(A, A).mean()) if len(A) > 1 else 0.0

def test_distributional_readout_restores_variance(synth):
    # Regression for the variance-collapse bug: a mean-field prediction is a near-degenerate
    # point cloud (tiny internal spread), which the *distributional* energy distance penalises
    # structurally regardless of whether the mean shift is right. Adding sampled control
    # residuals must restore the prediction's per-cell spread to ~ the observed niche scale.
    edges = build_knn_graph(synth, k=12)
    seed = TrivialSeed().fit(synth)
    base = GaussianProp().fit(synth, edges)
    learned = SimpleGCN(hidden=16, epochs=5).fit(synth, edges)
    resid = _control_residuals(synth)

    g_mean = fill_2x2(synth, "P0", edges, seed, base, learned, return_niches=True)
    g_dist = fill_2x2(synth, "P0", edges, seed, base, learned, return_niches=True, residuals=resid)
    obs = _niche_spread(g_mean["_niches"]["observed"])
    # mean-only GCN prediction collapses far below the observed spread...
    assert _niche_spread(g_mean["_niches"]["4"]) < 0.5 * obs
    # ...and the distributional readout brings it back into the observed ballpark.
    assert _niche_spread(g_dist["_niches"]["4"]) > 0.6 * obs

def test_baseline_column_does_not_leak(synth):
    # Regression: with GT seed, the Gaussian baseline must NOT reproduce the observed niche
    # (which happens if propagation starts from the observed matrix -> energy ~0). The harness
    # must start propagation from a CONTROL reference state, so cell (1) is clearly non-trivial.
    from spbench.config import run_benchmark
    res = run_benchmark(synth, perturbations=["P0"], k=8, gcn_kwargs={"hidden": 16, "epochs": 3})
    g = res["grids"]["P0"]
    assert g["1"]["energy_prop"] > 0.05, "baseline+GT-seed reproduced observed niche (leak)"
    assert res["leakage_pass"]["P0"]
