import numpy as np
from spbench.data import StandardData
from spbench.graph import build_knn_graph, neighbors_of
from spbench.permutation import permutation_null

def _spatial(n_per=40, planted=True, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per * 3
    coords = rng.normal(size=(n, 2)) * 5
    perturbation = np.array(["control"]*n)
    cen = rng.choice(n, n_per, replace=False)
    perturbation[cen] = "P0"
    X = rng.normal(size=(n, 10)).astype(float)
    d = StandardData(X=X, coords=coords, perturbation=perturbation,
                     cell_type=np.array(["A"]*n), batch=np.array(["b"]*n),
                     gene_names=[f"g{i}" for i in range(10)])
    edges = build_knn_graph(d, k=8)
    if planted:
        hit = set()
        for c in cen:
            for j in neighbors_of(c, edges):
                if perturbation[j] != "P0":
                    hit.add(int(j))
        idx = np.array(sorted(hit))
        if len(idx):
            X[idx] += 3.0
    return d, edges

def test_planted_perturbation_low_p_across_seeds():
    # strong planted niche shift -> consistently low p (no cherry-picked seed)
    ps = []
    for s in range(8):
        d, edges = _spatial(planted=True, seed=s)
        r = permutation_null(d, "P0", edges, n_perm=40, seed=0)
        assert np.isfinite(r["real"]) and r["null"]
        ps.append(r["p"])
    ps = np.array(ps)
    assert ps.mean() <= 0.15
    assert (ps <= 0.25).all()

def test_inert_perturbation_not_biased_low_across_seeds():
    # the OLD control-reference null pushed even inert perturbations to p~0.05-0.15 (false
    # significance). The two-sample relabeling null is exchangeable under H0 -> p ~uniform (mean
    # ~0.5). This locks the bias fix: inert p must NOT be systematically low.
    ps = []
    for s in range(8):
        d, edges = _spatial(planted=False, seed=s)
        r = permutation_null(d, "P0", edges, n_perm=40, seed=0)
        ps.append(r["p"])
    ps = np.array(ps)
    assert ps.mean() >= 0.30
    assert np.median(ps) >= 0.20

def test_empty_perturbation_returns_nan():
    d, edges = _spatial(planted=False, seed=3)
    r = permutation_null(d, "ABSENT", edges, n_perm=10, seed=0)
    assert np.isnan(r["p"])

def test_run_benchmark_exposes_perm():
    from spbench.config import run_benchmark
    d, edges = _spatial(planted=True, seed=5)
    res = run_benchmark(d, perturbations=["P0"], k=8, k_ref=4,
                        gcn_kwargs={"hidden": 8, "epochs": 3}, progress=False, n_perm=20)
    assert "perm" in res and "P0" in res["perm"]
    p = res["perm"]["P0"]["p"]
    assert (np.isnan(p)) or (0.0 <= p <= 1.0)
