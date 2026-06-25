import numpy as np
from spbench.data import StandardData
from spbench.graph import build_knn_graph
from spbench.harness import fill_2x2, _control_reference_aggregate, _control_residuals
from spbench.models.trivial_seed import TrivialSeed
from spbench.models.gaussian_prop import GaussianProp
from spbench.models.gcn_prop import SimpleGCN
from spbench.compare import evaluate_seed

def _planted(n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per * 3
    pert = np.array((["control"] * (2 * n_per)) + (["P0"] * n_per))
    X = rng.normal(scale=1.0, size=(n, 12))
    X[pert == "P0"] += np.array([2.0] + [0.0] * 11)
    d = StandardData(X=X, coords=rng.normal(size=(n, 2)), perturbation=pert,
                     cell_type=np.array(["A"] * n), batch=np.array(["b"] * n),
                     gene_names=[f"g{i}" for i in range(12)])
    return d

def _niches(d):
    edges = build_knn_graph(d, k=10)
    sm = TrivialSeed().fit(d); base = GaussianProp().fit(d, edges); lr = SimpleGCN(hidden=8, epochs=3).fit(d, edges)
    Xref = _control_reference_aggregate(d, edges); pool = _control_residuals(d)
    g = fill_2x2(d, "P0", edges, sm, base, lr, k_ref=5, X_ref=Xref, return_niches=True, residuals=pool)
    return g["_niches"]

def test_seed_pred_resid_restores_variance():
    n = _niches(_planted())
    sp = np.asarray(n["seed_pred"]); spr = np.asarray(n["seed_pred_resid"])
    assert sp.std(0).mean() < 1e-6
    assert spr.std(0).mean() > 0.3
    assert np.allclose(sp.mean(0), spr.mean(0), atol=0.25)

def test_seed_energy_fair_after_residual():
    n = _niches(_planted())
    r = evaluate_seed(n)
    m = float(np.mean(r["e_samples"]["model"])); nul = float(np.mean(r["e_samples"]["null"]))
    assert m < 2.0 * nul
    assert np.isfinite(r["pcc_delta"])

def test_pcc_delta_residual_invariant():
    n = _niches(_planted())
    r_raw = evaluate_seed({k: v for k, v in n.items() if k != "seed_pred_resid"})
    r_res = evaluate_seed(n)
    assert abs(r_raw["pcc_delta"] - r_res["pcc_delta"]) < 1e-9

def test_plot_delta_two_panels():
    import matplotlib; matplotlib.use("Agg")
    from spbench.config import run_benchmark
    from spbench.plotting import plot_delta, collect_delta
    d = _planted()
    res = run_benchmark(d, perturbations=["P0"], k=10, k_ref=5,
                        gcn_kwargs={"hidden": 8, "epochs": 3}, progress=False)
    sb, _ = collect_delta(res, "seed"); nb, _ = collect_delta(res, "niche")
    assert sb and nb
    fig = plot_delta(res)
    assert len(fig.axes) == 2
    titles = [a.get_title().lower() for a in fig.axes]
    assert any("seed" in t for t in titles) and any("niche" in t for t in titles)
