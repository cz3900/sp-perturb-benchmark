import numpy as np
import spbench.config as config
from spbench.config import run_benchmark
from spbench.synthetic import make_synthetic
from spbench.graph import build_knn_graph
from spbench.harness import _control_reference, _control_reference_aggregate

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
    # niche PCC-delta: oracle (perfect mean shift) ~ 1, every method present
    assert "pcc" in c and "model+learned" in c["pcc"]
    assert c["pcc"]["oracle"] > 0.9

def test_run_benchmark_uses_aggregate_control_reference(monkeypatch):
    # The active control reference in run_benchmark MUST be the aggregate path
    # (_control_reference_aggregate(data, edges)), NOT the legacy _control_reference(data).
    # We spy on the X_ref handed to fill_2x2 and on whether each helper was called.
    data = make_synthetic(seed=0)
    edges = build_knn_graph(data, k=8)
    expected_ref = _control_reference_aggregate(data, edges)   # the value the active path must use

    calls = {"aggregate": 0, "legacy": 0}

    def spy_aggregate(d, e):
        calls["aggregate"] += 1
        return _control_reference_aggregate(d, e)

    def spy_legacy(d):
        calls["legacy"] += 1
        return _control_reference(d)

    captured = {}
    real_fill = config.fill_2x2

    def spy_fill(*args, **kwargs):
        captured["X_ref"] = kwargs.get("X_ref")
        return real_fill(*args, **kwargs)

    monkeypatch.setattr(config, "_control_reference_aggregate", spy_aggregate)
    monkeypatch.setattr(config, "_control_reference", spy_legacy)
    monkeypatch.setattr(config, "fill_2x2", spy_fill)

    run_benchmark(data, perturbations=["P0"], k=8, k_ref=5,
                  gcn_kwargs={"hidden": 16, "epochs": 3}, progress=False, compare=False)

    # the aggregate helper is the active reference source; legacy is NOT invoked by the default path
    assert calls["aggregate"] >= 1, "run_benchmark did not build the aggregate control reference"
    assert calls["legacy"] == 0, "run_benchmark still calls the legacy _control_reference default"
    # and the X_ref actually fed into fill_2x2 equals the aggregate reference (not legacy-by-accident)
    assert captured["X_ref"] is not None
    np.testing.assert_allclose(captured["X_ref"], expected_ref, atol=1e-6)


def test_run_benchmark_scores_seed_directly():
    data = make_synthetic(seed=0)
    res = run_benchmark(data, perturbations=["P0", "P1"], k=8, k_ref=5,
                        gcn_kwargs={"hidden": 16, "epochs": 3}, progress=False)
    s = res["seed"]["P0"]
    # seed gets its own PCC-delta (direction) and MSE (magnitude), independent of the niche
    assert "pcc_delta" in s and "mse" in s
    assert s["mse"] >= 0.0
    assert -1.0 <= s["pcc_delta"] <= 1.0
