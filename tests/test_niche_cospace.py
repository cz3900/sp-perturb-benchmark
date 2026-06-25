"""Part B: when eval_X is the (n_cells, G) scoring matrix (scGEN log-norm), fill_2x2 must build the
WHOLE niche path in that space — observed/reference niche, X_ref, residuals, GT seed — so a model
whose seed lives there (scGEN) is scored consistently instead of space-mixed."""
import numpy as np
from spbench.graph import build_knn_graph
from spbench.harness import fill_2x2
from spbench.models.gaussian_prop import GaussianProp
from spbench.compare import compare_to_baseline


def test_fill_2x2_niche_cospace(synth):
    data = synth
    edges = build_knn_graph(data, k=6)
    p = "P0"
    centers = np.where(data.perturbation == p)[0]
    assert len(centers) > 0
    SHIFT = 100.0
    evalX = np.asarray(data.X, float) + SHIFT            # a distinct scoring space (pure additive offset)

    class PerCenterSeed:                                  # scGEN-like loader: per-center cached rows in eval_X space
        def centers(self, pert): return centers
        def predict_seed(self, pert, X): return evalX[centers]

    class DataXSeed(PerCenterSeed):                       # the SAME seed but in data.X space (no offset)
        def predict_seed(self, pert, X): return np.asarray(data.X, float)[centers]

    gp = GaussianProp().fit(data, edges)
    grid_e = fill_2x2(data, p, edges, PerCenterSeed(), gp, gp, return_niches=True, eval_X=evalX)   # co-spaced
    grid_x = fill_2x2(data, p, edges, DataXSeed(), gp, gp, return_niches=True)                     # all data.X
    ne, nx = grid_e["_niches"], grid_x["_niches"]

    # whole niche path co-spaced into eval_X: every component is EXACTLY the data.X version + SHIFT
    for key in ("observed", "reference", "seed_obs", "1", "2", "3", "4"):
        assert np.allclose(ne[key], nx[key] + SHIFT), key
    assert ne["eval_X"] is None                                       # transform consumed (already co-spaced)

    # niche scoring works on the co-spaced niche; PCC-delta is shift-invariant -> identical to data.X
    out_e = compare_to_baseline(ne)
    out_x = compare_to_baseline(nx)
    assert np.isfinite(out_e["pcc"]["model+base"]) and np.isfinite(out_e["mag"]["model+base"])
    assert np.isclose(out_e["pcc"]["model+base"], out_x["pcc"]["model+base"])
