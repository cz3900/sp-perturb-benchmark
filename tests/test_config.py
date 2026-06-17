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

def test_run_benchmark_compares_to_baseline():
    data = make_synthetic(seed=0)
    res = run_benchmark(data, perturbations=["P0", "P1"], k=8, k_ref=5,
                        gcn_kwargs={"hidden": 16, "epochs": 3}, progress=False)
    c = res["compare"]["P0"]
    # every 2x2 cell + the baseline have an energy distance and a gain (= e_null - e)
    for m in ("GT+base", "GT+learned", "model+base", "model+learned", "null"):
        assert m in c["e"] and m in c["gain"]
    assert c["gain"]["null"] == 0.0
    assert abs(c["gain"]["model+learned"] - (c["e"]["null"] - c["e"]["model+learned"])) < 1e-9
