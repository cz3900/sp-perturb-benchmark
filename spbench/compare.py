"""Compare every prediction to the no-effect baseline, in one common currency.

This is the ORCHESTRATION layer — it does NOT define metrics. The metric primitives live under
spbench.metrics (PCC-delta, MSE) and are pulled from the registry. Here we only assemble them: the
no-effect baseline (`null`), the per-method PCC-delta (direction) and relative magnitude (`mag`),
and the seed/niche scores.

Scoring is mean-based and uses the FULL sample (no matched-n subsampling): PCC-delta correlates the
predicted gene-wise mean-shift with the observed one, and `mag` is the relative size of that shift.
The four 2x2 cells are predictions; `null` is the laziest prediction ('the neighbours did not
change' = the aggregate same-type control niche). `pcc['null']` is NaN by construction (a flat
delta has no direction to correlate) — a sanity check.
"""
import numpy as np
from .metrics import get_metric

# 2x2 cell key -> human label (seed source + propagation model)
METHODS = {"1": "GT+base", "2": "GT+learned", "3": "model+base", "4": "model+learned"}
DEPLOYABLE = "model+learned"   # the cell you would actually ship (predicted seed + learned prop)


def _apply_eval_X(A, eval_X):
    """Push a cell array into the unified scoring space. eval_X=None -> identity (old behavior,
    no copy). eval_X is a callable applied array-wise (default choice np.arcsinh) so that
    pred / obs / ref are all compared in ONE common space — pcc_delta is not cross-space robust,
    so its delta baseline (reference) must share the same transform as pred and obs. np.arcsinh
    (not np.log1p) because the expression matrices are signed and log1p would NaN on values < -1."""
    A = np.asarray(A, float)
    return A if eval_X is None else np.asarray(eval_X(A), float)


def _mag(pred_x, obs_x, ref_x):
    """Relative magnitude of the predicted mean-shift vs the true mean-shift, in the scoring space:
    ||mean(pred)-mean(ref)|| / ||mean(obs)-mean(ref)||. 1 = right size, <1 under-, >1 over-shoot.
    nan when the true shift is ~0 (nothing to scale against) or an input is empty."""
    pred_x = np.asarray(pred_x, float); obs_x = np.asarray(obs_x, float); ref_x = np.asarray(ref_x, float)
    if len(pred_x) == 0 or len(obs_x) == 0 or len(ref_x) == 0:
        return float("nan")
    rm = ref_x.mean(0)
    dt = float(np.linalg.norm(obs_x.mean(0) - rm))
    if not np.isfinite(dt) or dt < 1e-12:
        return float("nan")
    return float(np.linalg.norm(pred_x.mean(0) - rm) / dt)


def evaluate_seed(niches, eval_X=None):
    """Direct seed score: MODEL seed vs observed perturbed cells. Returns PCC-delta (direction,
    baseline=matched control), MSE (magnitude), and `mag` (relative shift size). All mean-based on
    the full sample (no subsampling).

    eval_X (callable | None): unified scoring-space transform applied to pred/obs/ref before
    scoring (pred/obs/ref must share one transform; pcc_delta is not cross-space robust)."""
    obs = _apply_eval_X(niches.get("seed_obs", np.zeros((0, 0))), eval_X)
    pred = _apply_eval_X(niches.get("seed_pred", np.zeros((0, 0))), eval_X)
    ref = _apply_eval_X(niches.get("seed_ref", np.zeros((0, 0))), eval_X)
    if len(obs) == 0 or len(pred) == 0 or len(ref) == 0:
        return {"pcc_delta": float("nan"), "mse": float("nan"), "mag": float("nan"),
                "n": int(len(obs))}
    return {"pcc_delta": get_metric("pcc_delta").compute(pred, obs, {"reference": ref}),
            "mse": get_metric("mse").compute(pred, obs), "mag": _mag(pred, obs, ref),
            "n": int(len(obs))}


def compare_to_baseline(niches, extra=None, eval_X=None):
    """Niche PCC-delta (direction) and relative magnitude (`mag`) of every 2x2 cell and the
    no-effect baseline (`null`) vs the observed niche, mean-based on the full sample.

    niches    : dict with 'observed', 'reference', and the four cell arrays '1'..'4'.
    extra     : optional {name: predicted-niche array} for external models (e.g. {'CONCERT': arr}),
                scored on exactly the same PCC-delta / mag footing as the 2x2 cells.

    Returns {'pcc': {method: PCC-delta of the niche shift}, 'mag': {method: relative shift size},
    'n': observed sample size}. pcc['null'] is NaN (flat shift) — a sanity check. PCC-delta is
    mean-based, bounded and self-anchored, so it is robust to weak signal and variance scale.
    """
    obs = np.asarray(niches["observed"], float)
    ref = np.asarray(niches["reference"], float)
    clouds = {METHODS[k]: np.asarray(niches[k], float) for k in METHODS if k in niches}
    clouds["null"] = ref
    if extra:                                                  # external models (e.g. CONCERT)
        for nm, arr in extra.items():
            clouds[nm] = np.asarray(arr, float)
    # PCC-delta of the niche shift per method, scored in the unified eval_X space: pred/obs/ref must
    # share one transform (pcc_delta is not cross-space robust).
    pccm = get_metric("pcc_delta")
    obs_x = _apply_eval_X(obs, eval_X)
    ref_x = _apply_eval_X(ref, eval_X)
    pcc, mag = {}, {}
    for k, c in clouds.items():
        cx = _apply_eval_X(c, eval_X)
        pcc[k] = pccm.compute(cx, obs_x, {"reference": ref_x})
        mag[k] = _mag(cx, obs_x, ref_x)
    return {"pcc": pcc, "mag": mag, "n": int(len(obs))}
