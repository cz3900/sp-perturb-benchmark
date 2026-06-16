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
