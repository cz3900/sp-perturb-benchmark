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


def compare_to_baseline(niches, residuals=None, repeats=20, seed=0, max_n=300):
    """Matched-n energy distance of every 2x2 cell, the no-effect baseline, and an oracle ceiling
    to the observed niche, plus gain = e_null - e_method.

    niches    : dict with 'observed', 'reference', and the four cell arrays '1'..'4'.
    residuals : per-cell-type control residual pools (for the oracle ceiling); None -> no oracle.

    Returns {'e': {method: dist}, 'gain': {method: e_null - dist}, 'n': matched size,
             'has_effect': bool}. gain['null'] is 0 by construction (the baseline line).
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
    # 'real effect' = the no-effect baseline is itself clearly far from observed (vs the oracle floor)
    floor = e.get("oracle", 0.0)
    has_effect = bool(null > 2 * floor) if "oracle" in e else bool(null > 0)
    return {"e": e, "gain": gain, "n": n, "has_effect": has_effect}
