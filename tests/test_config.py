import numpy as np
from spbench.config import run_benchmark
from spbench.synthetic import make_synthetic

def test_run_benchmark_returns_grids_and_attribution():
    data = make_synthetic(seed=0)
    res = run_benchmark(data, perturbations=["P0", "P1"], k=8, k_ref=5,
                        gcn_kwargs={"hidden": 16, "epochs": 3})
    assert "P0" in res["grids"] and "P1" in res["grids"]
    assert "end_to_end" in res["attribution"]["P0"]
    assert isinstance(res["leakage_pass"]["P0"], (bool, np.bool_))
