# tests/test_compare_eval_x.py
import numpy as np
import pytest
from spbench.compare import compare_to_baseline


def _niches():
    rng = np.random.default_rng(1)
    G = 5
    ref = rng.random((40, G)) * 4.0
    obs = ref.mean(0) + 2.0 + rng.random((40, G)) * 0.2     # clear shift
    c1 = ref.mean(0) + 1.8 + rng.random((40, G)) * 0.2      # good prediction
    c2 = ref.mean(0) + 0.2 + rng.random((40, G)) * 0.2      # weak prediction
    return {"observed": obs, "reference": ref,
            "1": c1, "2": c2, "3": c1.copy(), "4": c2.copy()}


def test_compare_default_unchanged():
    out = compare_to_baseline(_niches(), repeats=3, seed=0)
    assert set(out) == {"e", "gain", "pcc", "mag", "n", "has_effect", "e_samples"}
    assert "null" in out["e"] and out["gain"]["null"] == pytest.approx(0.0, abs=1e-9)
    for k in ["GT+base", "GT+learned", "model+base", "model+learned"]:
        assert np.isfinite(out["pcc"][k])


def test_eval_x_preserves_structure_and_keys():
    base = compare_to_baseline(_niches(), repeats=3, seed=0)
    logged = compare_to_baseline(_niches(), repeats=3, seed=0, eval_X=np.arcsinh)
    assert set(logged["e"]) == set(base["e"])
    assert set(logged["pcc"]) == set(base["pcc"])
    # energy/gain space is untouched by eval_X -> identical numbers
    for k in base["e"]:
        assert logged["e"][k] == pytest.approx(base["e"][k], rel=1e-9, abs=1e-9)


def test_eval_x_changes_pcc_only():
    base = compare_to_baseline(_niches(), repeats=3, seed=0)
    logged = compare_to_baseline(_niches(), repeats=3, seed=0, eval_X=np.arcsinh)
    # PCC-delta recomputed in arcsinh space -> finite, and differs from raw-space PCC
    assert np.isfinite(logged["pcc"]["GT+base"])
    diffs = [abs(logged["pcc"][k] - base["pcc"][k])
             for k in ["GT+base", "GT+learned", "model+base", "model+learned"]]
    assert max(diffs) > 1e-6


def test_eval_x_with_extra_model():
    n = _niches()
    extra = {"CONCERT": n["1"].copy()}
    out = compare_to_baseline(n, repeats=3, seed=0, eval_X=np.arcsinh, extra=extra)
    assert "CONCERT" in out["pcc"] and np.isfinite(out["pcc"]["CONCERT"])
    assert "CONCERT" in out["e"]
