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
    out = compare_to_baseline(_niches())
    assert set(out) == {"pcc", "mag", "n"}
    assert "null" in out["pcc"] and np.isnan(out["pcc"]["null"])   # flat shift -> no direction
    for k in ["GT+base", "GT+learned", "model+base", "model+learned"]:
        assert np.isfinite(out["pcc"][k])


def test_eval_x_preserves_structure_and_keys():
    base = compare_to_baseline(_niches())
    logged = compare_to_baseline(_niches(), eval_X=np.arcsinh)
    assert set(logged["pcc"]) == set(base["pcc"])
    assert set(logged["mag"]) == set(base["mag"])


def test_eval_x_changes_pcc_only():
    base = compare_to_baseline(_niches())
    logged = compare_to_baseline(_niches(), eval_X=np.arcsinh)
    # PCC-delta recomputed in arcsinh space -> finite, and differs from raw-space PCC
    assert np.isfinite(logged["pcc"]["GT+base"])
    diffs = [abs(logged["pcc"][k] - base["pcc"][k])
             for k in ["GT+base", "GT+learned", "model+base", "model+learned"]]
    assert max(diffs) > 1e-6


def test_eval_x_with_extra_model():
    n = _niches()
    extra = {"CONCERT": n["1"].copy()}
    out = compare_to_baseline(n, eval_X=np.arcsinh, extra=extra)
    assert "CONCERT" in out["pcc"] and np.isfinite(out["pcc"]["CONCERT"])
    assert "CONCERT" in out["mag"]
