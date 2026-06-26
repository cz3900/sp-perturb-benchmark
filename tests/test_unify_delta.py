import numpy as np
from spbench.data import StandardData
from spbench.graph import build_knn_graph
from spbench.harness import fill_2x2, _control_reference_aggregate
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
    Xref = _control_reference_aggregate(d, edges)
    g = fill_2x2(d, "P0", edges, sm, base, lr, k_ref=5, X_ref=Xref, return_niches=True)
    return g["_niches"]

def test_seed_pcc_delta_recovers_direction():
    n = _niches(_planted())
    r = evaluate_seed(n)
    # the planted seed shift is gene 0 only; the model seed must recover that direction
    assert np.isfinite(r["pcc_delta"])
    assert r["pcc_delta"] > 0.5

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
