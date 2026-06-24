from spbench.adapters.shen import _map_shen_perturbation


def test_shen_perturbation_mapping():
    vals = ["C9orf72", "Doublet", "Non-perturbed", "msafe", "lrrk2", "gfap"]
    out = list(_map_shen_perturbation(vals))
    # gene KOs kept; Doublet + Non-perturbed -> 'none' (Plan-2 control pool); msafe (safe-harbor) -> 'control'
    assert out == ["C9orf72", "none", "none", "control", "lrrk2", "gfap"]
