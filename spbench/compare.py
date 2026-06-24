"""Compare every prediction to the no-effect baseline, in one common currency.

This is the ORCHESTRATION layer — it does NOT define metrics. The metric primitives live under
spbench.metrics (energy distance, PCC-delta, MSE) and are pulled from the registry / the shared
`energy_distance` primitive. Here we only assemble them: matched-n subsampling, the no-effect
baseline (`e_null`) and oracle ceiling, the gain, and the seed/niche scores.

Each entry is an energy distance `e` between a *prediction* and the *observed* perturbed niche
(small e = the prediction is close to reality). The four 2x2 cells are predictions; `null` is the
laziest prediction ('the neighbours did not change' = the aggregate same-type control niche); `oracle` is the
best a non-leaking model could do (perfect mean shift + control-population variance) = a ceiling.

The headline quantity is the *gain* over the baseline:

    gain = e_null - e_method        (>0  the method beats 'no effect';  <0  worse than doing nothing)

All distances in one perturbation share the SAME observed subsample on every repeat and a common
matched sample size n, so the finite-sample bias of the energy distance cancels and `gain` is a
clean paired difference.
"""
import numpy as np
from .metrics import get_metric
from .metrics.energy import energy_distance

# 2x2 cell key -> human label (seed source + propagation model)
METHODS = {"1": "GT+base", "2": "GT+learned", "3": "model+base", "4": "model+learned"}
DEPLOYABLE = "model+learned"   # the cell you would actually ship (predicted seed + learned prop)


def _sub(A, n, rng):
    return A if len(A) <= n else A[rng.choice(len(A), n, replace=False)]


def _apply_eval_X(A, eval_X):
    """Push a cell array into the unified scoring space. eval_X=None -> identity (old behavior,
    no copy). eval_X is a callable applied array-wise (default choice np.arcsinh) so that
    pred / obs / ref are all compared in ONE common space — pcc_delta is not cross-space robust,
    so its delta baseline (reference) must share the same transform as pred and obs. np.arcsinh
    (not np.log1p) because the expression matrices are signed and log1p would NaN on values < -1."""
    A = np.asarray(A, float)
    return A if eval_X is None else np.asarray(eval_X(A), float)


def evaluate_seed(niches, eval_X=None):
    """Direct seed score (decoupled from propagation): the MODEL seed (predicted perturbed-cell
    expression) vs the observed perturbed cells. PCC-delta = direction of the gene-wise shift
    (baseline = matched control cells); MSE = magnitude of the mean error. Both metrics come from
    the registry.

    eval_X (callable | None): unified scoring-space transform applied to pred/obs/ref before
    scoring (default choice np.arcsinh), so all three live in the same variance-stabilized space
    (pcc_delta is not cross-space robust). eval_X=None keeps the old raw-space behavior."""
    obs = _apply_eval_X(niches.get("seed_obs", np.zeros((0, 0))), eval_X)
    pred = _apply_eval_X(niches.get("seed_pred", np.zeros((0, 0))), eval_X)
    ref = _apply_eval_X(niches.get("seed_ref", np.zeros((0, 0))), eval_X)
    if len(obs) == 0 or len(pred) == 0 or len(ref) == 0:
        return {"pcc_delta": float("nan"), "mse": float("nan"), "n": int(len(obs))}
    return {"pcc_delta": get_metric("pcc_delta").compute(pred, obs, {"reference": ref}),
            "mse": get_metric("mse").compute(pred, obs), "n": int(len(obs))}


def compare_to_baseline(niches, residuals=None, repeats=20, seed=0, max_n=300, extra=None, eval_X=None):
    """Matched-n energy distance of every 2x2 cell, the no-effect baseline, and an oracle ceiling
    to the observed niche, plus gain = e_null - e_method and a niche PCC-delta per method.

    niches    : dict with 'observed', 'reference', and the four cell arrays '1'..'4'.
    residuals : per-cell-type control residual pools (for the oracle ceiling); None -> no oracle.
    extra     : optional {name: predicted-niche array} for external models (e.g. {'CONCERT': arr}),
                scored on exactly the same matched-n / gain / PCC-delta footing as the 2x2 cells.

    Returns {'e': {method: dist}, 'gain': {method: e_null - dist}, 'pcc': {method: PCC-delta of
    the niche shift}, 'n': matched size, 'has_effect': bool, 'e_samples': {method: per-repeat
    energy list whose nanmean is 'e'} (box-plot data)}. gain['null'] is 0 by construction
    (the baseline line); pcc['oracle'] is ~1 and pcc['null'] is NaN (flat shift), both sanity
    checks. PCC-delta complements the energy distance: it is mean-based, bounded and self-anchored,
    so it is robust where the energy distance is fragile (weak signal, variance scale).
    """
    obs = np.asarray(niches["observed"], float)
    ref = np.asarray(niches["reference"], float)
    clouds = {METHODS[k]: np.asarray(niches[k], float) for k in METHODS if k in niches}
    clouds["null"] = ref
    if extra:                                                  # external models (e.g. CONCERT)
        for nm, arr in extra.items():
            clouds[nm] = np.asarray(arr, float)
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
            acc[k].append(energy_distance(_sub(c, n, rng), O))
    e = {k: float(np.nanmean(v)) for k, v in acc.items()}
    null = e["null"]
    gain = {k: null - e[k] for k in clouds}
    # PCC-delta of the niche shift per method, scored in the unified eval_X space: pred/obs/ref must
    # share one transform (pcc_delta is not cross-space robust). energy/gain above stay in raw space
    # (residual variance is calibrated there; matched-n bias-cancellation must not move space).
    pccm = get_metric("pcc_delta")
    obs_x = _apply_eval_X(obs, eval_X)
    ref_x = _apply_eval_X(ref, eval_X)
    pcc = {k: pccm.compute(_apply_eval_X(c, eval_X), obs_x, {"reference": ref_x})
           for k, c in clouds.items()}
    # 'real effect' = the no-effect baseline is itself clearly far from observed (vs the oracle floor)
    floor = e.get("oracle", 0.0)
    has_effect = bool(null > 2 * floor) if "oracle" in e else bool(null > 0)
    return {"e": e, "gain": gain, "pcc": pcc, "n": n, "has_effect": has_effect,
            "e_samples": {k: list(v) for k, v in acc.items()}}
