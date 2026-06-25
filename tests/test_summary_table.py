import numpy as np
from spbench.plotting import summary_table

def _fake_res():
    # mimic run_benchmark output shape for two guides
    return {
        "seed": {
            "GeneA": {"pcc_delta": 0.8},
            "GeneB": {"pcc_delta": float("nan")},
        },
        "compare": {
            "GeneA": {"pcc": {"model+base": 0.5, "model+learned": 0.6, "CONCERT": 0.7, "null": float("nan")}},
            "GeneB": {"pcc": {"model+base": 0.1, "model+learned": 0.2, "CONCERT": 0.3, "null": float("nan")}},
        },
    }

def test_summary_table_has_seed_and_niche_columns():
    rows = summary_table(_fake_res())
    by_guide = {r["guide"]: r for r in rows}
    assert set(by_guide) == {"GeneA", "GeneB"}
    a = by_guide["GeneA"]
    assert a["seed_pcc_delta"] == 0.8
    # niche columns: deployable cells + external methods, NOT null
    assert a["niche_Gaussian"] == 0.5
    assert a["niche_GCN"] == 0.6
    assert a["niche_CONCERT"] == 0.7
    assert "niche_null" not in a

def test_summary_table_propagates_nan():
    rows = summary_table(_fake_res())
    b = {r["guide"]: r for r in rows}["GeneB"]
    assert np.isnan(b["seed_pcc_delta"])
