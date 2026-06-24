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

def test_planted_perturbation_has_low_p():
    d, edges = _spatial(planted=True, seed=1)
    r = permutation_null(d, "P0", edges, n_perm=40, seed=0)
    assert np.isfinite(r["real"]) and r["null"]
    assert r["p"] <= 0.2

def test_inert_perturbation_has_high_p():
    d, edges = _spatial(planted=False, seed=2)
    r = permutation_null(d, "P0", edges, n_perm=40, seed=0)
    assert r["p"] >= 0.2

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
