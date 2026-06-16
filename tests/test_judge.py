from spbench.judge import attribute, leakage_pass

def test_attribution_signs():
    grid = {"1": {"energy_prop": 1.0}, "2": {"energy_prop": 0.6},
            "3": {"energy_prop": 1.8}, "4": {"energy_prop": 1.2}}
    a = attribute(grid)
    assert a["seed_cost"] > 0
    assert a["learned_value"] > 0
    assert a["end_to_end"] == 1.2

def test_leakage_flags_suspicious_cell2():
    assert leakage_pass({"2": {"energy_prop": 1e-6}}, floor=1e-3) is False
    assert leakage_pass({"2": {"energy_prop": 0.5}}, floor=1e-3) is True
