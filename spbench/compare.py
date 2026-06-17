"""Compare every prediction to the no-effect baseline, in one common currency.

Each entry is an energy distance `e` between a *prediction* and the *observed* perturbed niche
(small e = the prediction is close to reality). The four 2x2 cells are predictions; `null` is the
laziest prediction ('the neighbours did not change' = the matched control niche); `oracle` is the
best a non-leaking model could do (perfect mean shift + control-population variance) and acts as a
ceiling.

The headline quantity is the *gain* over the baseline:

    gain = e_null - e_method        (>0  the method beats 'no effect';  <0  worse than doing nothing)

All distances in one perturbation share the SAME observed subsample on every repeat and a common
matched sample size n, so the finite-sample bias of the energy distance cancels and `gain` is a
clean paired difference. No ratios, no calibration.
"""
import numpy as np
from scipy.spatial.distance import cdist

# 2x2 cell key -> human label (seed source + propagation model)
METHODS = {"1": "GT+base", "2": "GT+learned", "3": "model+base", "4": "model+learned"}
DEPLOYABLE = "model+learned"   # the cell you would actually ship (predicted seed + learned prop)


def _edist(A, B):
    A = np.asarray(A, float); B = np.asarray(B, float)
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    return float(max(0.0, 2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean()))


def _sub(A, n, rng):
    return A if len(A) <= n else A[rng.choice(len(A), n, replace=False)]


def _pcc_delta(d_pred, d_true):
    """Pearson correlation between the predicted and true gene-wise shift vectors (the 'delta' =
    perturbed mean - control mean). Captures whether the prediction moves the right genes in the
    right direction (GEARS/scGPT convention). Bounded [-1, 1], self-anchored at 0 (a no-effect
    prediction has a flat delta -> NaN -> no skill). NaN if either shift is essentially flat."""
    d_pred = np.asarray(d_pred, float); d_true = np.asarray(d_true, float)
    if d_pred.std() < 1e-12 or d_true.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(d_pred, d_true)[0, 1])


def _mse(a, b):
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


def evaluate_seed(niches):
    """Direct seed score (decoupled from propagation): the MODEL seed (predicted perturbed-cell
    expression) vs the observed perturbed cells. PCC-delta = direction of the gene-wise shift
    (baseline = matched control cells); MSE = magnitude of the mean error. Both are mean-based,
    so they need no distributional readout and have none of the energy distance's fragility."""
    obs = np.asarray(niches.get("seed_obs", np.zeros((0, 0))), float)
    pred = np.asarray(niches.get("seed_pred", np.zeros((0, 0))), float)
    ref = np.asarray(niches.get("seed_ref", np.zeros((0, 0))), float)
    if len(obs) == 0 or len(pred) == 0 or len(ref) == 0:
        return {"pcc_delta": float("nan"), "mse": float("nan"), "n": int(len(obs))}
    ref_mean = ref.mean(0)
    return {"pcc_delta": _pcc_delta(pred.mean(0) - ref_mean, obs.mean(0) - ref_mean),
            "mse": _mse(pred.mean(0), obs.mean(0)), "n": int(len(obs))}


def compare_to_baseline(niches, residuals=None, repeats=20, seed=0, max_n=300):
    """Matched-n energy distance of every 2x2 cell, the no-effect baseline, and an oracle ceiling
    to the observed niche, plus gain = e_null - e_method.

    niches    : dict with 'observed', 'reference', and the four cell arrays '1'..'4'.
    residuals : per-cell-type control residual pools (for the oracle ceiling); None -> no oracle.

    Returns {'e': {method: dist}, 'gain': {method: e_null - dist}, 'pcc': {method: PCC-delta of
    the niche shift}, 'n': matched size, 'has_effect': bool}. gain['null'] is 0 by construction
    (the baseline line); pcc['oracle'] is ~1 and pcc['null'] is NaN (flat shift), both sanity
    checks. PCC-delta complements the energy distance: it is mean-based, bounded and self-anchored,
    so it is robust where the energy distance is fragile (weak signal, variance scale).
    """
    obs = np.asarray(niches["observed"], float)
    ref = np.asarray(niches["reference"], float)
    clouds = {METHODS[k]: np.asarray(niches[k], float) for k in METHODS if k in niches}
    clouds["null"] = ref
    if residuals is not None and len(obs) > 1:                 # oracle ceiling (best non-leaking)
        pool = residuals[None]
        r0 = np.random.default_rng(seed)
        clouds["oracle"] = obs.mean(0) + pool[r0.integers(len(pool), size=len(obs))]

    sizes = [len(obs)] + [len(c) for c in clouds.values()]
    n = max(2, min(min(sizes), max_n))
    rng = np.random.default_rng(seed + 1)
    acc = {k: [] for k in clouds}
    for _ in range(repeats):
        O = _sub(obs, n, rng)                                  # one observed draw shared by all
        for k, c in clouds.items():
            acc[k].append(_edist(_sub(c, n, rng), O))
    e = {k: float(np.nanmean(v)) for k, v in acc.items()}
    null = e["null"]
    gain = {k: null - e[k] for k in clouds}
    # PCC-delta of the niche shift per method (direction of the gene-wise change vs the true shift)
    ref_mean = ref.mean(0)
    d_true = obs.mean(0) - ref_mean
    pcc = {k: _pcc_delta(np.asarray(c, float).mean(0) - ref_mean, d_true) for k, c in clouds.items()}
    # 'real effect' = the no-effect baseline is itself clearly far from observed (vs the oracle floor)
    floor = e.get("oracle", 0.0)
    has_effect = bool(null > 2 * floor) if "oracle" in e else bool(null > 0)
    return {"e": e, "gain": gain, "pcc": pcc, "n": n, "has_effect": has_effect}
