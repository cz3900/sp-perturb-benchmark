import numpy as np
from spbench.data import StandardData
from spbench.reference_aggregate import control_reference_centers
from spbench.config import run_benchmark

def _data(perts, cts, n_genes=12, seed=0):
    rng = np.random.default_rng(seed)
    n = len(perts)
    X = rng.normal(size=(n, n_genes)).astype(float)
    coords = rng.normal(size=(n, 2))
    return StandardData(X=X, coords=coords, perturbation=np.array(perts),
                        cell_type=np.array(cts), batch=np.array(["b"]*n),
                        gene_names=[f"g{i}" for i in range(n_genes)])

def test_control_pool_uses_control_when_present():
    d = _data(["control","control","P0","P0"], ["A","A","A","A"])
    assert d.has_ntc is True
    assert np.array_equal(d.control_pool, d.is_control)

def test_control_pool_falls_back_to_none_when_no_ntc():
    d = _data(["none","none","P0","P0"], ["A","A","A","A"])
    assert d.has_ntc is False
    assert np.array_equal(d.control_pool, d.is_unlabeled)

def test_single_cell_type_degenerates_to_all_controls():
    d = _data(["control"]*4 + ["P0","P0"], ["A"]*6)
    centers = np.where(d.perturbation == "P0")[0]
    refs = control_reference_centers(d, centers)
    ctrl_idx = np.where(d.is_control)[0]
    for r in refs:
        assert np.array_equal(np.sort(r), np.sort(ctrl_idx))

def test_run_benchmark_no_ntc_non_nan():
    rng = np.random.default_rng(1)
    n = 60
    perts = np.array((["none"]*30) + (["P0"]*30))
    cts = np.array(["A"]*n)
    X = rng.normal(size=(n, 12)); X[perts=="P0"] += 1.5
    coords = rng.normal(size=(n, 2))
    d = StandardData(X=X, coords=coords, perturbation=perts, cell_type=cts,
                     batch=np.array(["b"]*n), gene_names=[f"g{i}" for i in range(12)])
    res = run_benchmark(d, perturbations=["P0"], k=8, k_ref=4,
                        gcn_kwargs={"hidden": 8, "epochs": 3}, progress=False)
    s = res["seed"]["P0"]; c = res["compare"]["P0"]
    assert np.isfinite(s["pcc_delta"])
    assert np.isfinite(c["e"]["model+base"])
