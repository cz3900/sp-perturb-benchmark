from spbench.judge import attribute, leakage_pass

def test_attribution_signs():
    # higher pcc_prop = better. GT cells (1,2) beat model cells (3,4); learned (2) beats base (1).
    grid = {"1": {"pcc_prop": 0.6}, "2": {"pcc_prop": 0.8},
            "3": {"pcc_prop": 0.3}, "4": {"pcc_prop": 0.5}}
    a = attribute(grid)
    assert a["seed_cost"] > 0          # GT seed (1) better than model seed (3)
    assert a["learned_value"] > 0      # learned (2) better than baseline (1)
    assert a["end_to_end"] == 0.5

def test_leakage_flags_suspicious_cell2():
    # PCC-delta ~1 = propagation reproduced the observed niche -> leakage
    assert leakage_pass({"2": {"pcc_prop": 1.0 - 1e-6}}, ceiling=1.0 - 1e-3) is False
    assert leakage_pass({"2": {"pcc_prop": 0.5}}, ceiling=1.0 - 1e-3) is True
