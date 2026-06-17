import numpy as np
from spbench.calibrate import calibrate_edistance, skill_score, propagation_skill

def _draw(mean, n=200, d=10, sd=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, sd, size=(n, d))

def test_no_signal_when_same_distribution():
    # observed and reference from the SAME distribution -> S ~ floor -> no signal
    obs = _draw(0.0, seed=1)
    ref = _draw(0.0, seed=2)
    cal = calibrate_edistance(obs, ref, repeats=15)
    assert not cal["has_signal"]

def test_signal_when_shifted():
    obs = _draw(1.5, seed=1)   # clear shift vs reference
    ref = _draw(0.0, seed=2)
    cal = calibrate_edistance(obs, ref, repeats=15)
    assert cal["has_signal"]
    assert cal["S"] > cal["floor"]

def test_skill_perfect_prediction_is_about_one():
    obs = _draw(1.5, seed=1)
    ref = _draw(0.0, seed=2)
    # predict from the SAME distribution as observed -> error ~ floor -> skill ~ 1
    out = propagation_skill(predicted=_draw(1.5, seed=3), observed=obs, reference=ref, repeats=15)
    assert out["has_signal"] and out["skill"] > 0.7

def test_skill_no_effect_prediction_is_about_zero():
    obs = _draw(1.5, seed=1)
    ref = _draw(0.0, seed=2)
    # predict the control (no effect) -> error ~ S -> skill ~ 0
    out = propagation_skill(predicted=_draw(0.0, seed=3), observed=obs, reference=ref, repeats=15)
    assert out["has_signal"] and out["skill"] < 0.4

def test_skill_above_floor_is_clipped_and_flagged():
    cal = {"has_signal": True, "S": 5.0, "floor": 1.0}
    s = skill_score(model_error=0.5, cal=cal)   # 0.5 < floor -> raw > 1
    assert s["clipped"] and s["skill"] == 1.0 and s["raw"] > 1.0

def test_no_signal_skill_is_nan():
    cal = {"has_signal": False, "S": 5.0, "floor": 4.9}
    s = skill_score(model_error=4.95, cal=cal)
    assert np.isnan(s["skill"]) and s["has_signal"] is False
