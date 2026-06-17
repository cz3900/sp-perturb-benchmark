"""Calibrate the propagation E-distance scale so an absolute E-distance (~6, background
dominated) becomes an interpretable 0..1 skill score.

For one perturbation's propagation:
  floor = E-distance between two random halves of the OBSERVED perturbed niche
          (pure sampling noise; the best score a perfect prediction could reach)
  S     = E-distance(observed perturbed niche, control/reference niche)
          (the ceiling = total real effect there is to predict)
  skill = (S - model_error) / (S - floor)
          0 = no better than predicting the control niche (no effect),
          1 = perfect (at the noise floor).

All distances use a MATCHED sample size n so the finite-sample bias of the energy distance
is comparable across comparisons. When S is not clearly above the floor the perturbation has
no reliable signal and the skill score is left undefined (NaN). skill > 1 (model_error below
the noise floor) is clipped to 1 and flagged.
"""
import numpy as np
from scipy.spatial.distance import cdist


def _edist(A, B):
    """Raw energy distance 2*mean||A-B|| - mean||A-A'|| - mean||B-B'|| on the given arrays."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if len(A) == 0 or len(B) == 0:
        return float("nan")
    return float(2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())


def _sub(A, n, rng):
    if len(A) <= n:
        return A
    return A[rng.choice(len(A), n, replace=False)]


def edist_matched(A, B, n, repeats=20, seed=0):
    """Mean +/- std energy distance between A and B, each independently subsampled to size n
    on every repeat (matched sample sizes -> comparable finite-sample bias)."""
    rng = np.random.default_rng(seed)
    vals = np.array([_edist(_sub(A, n, rng), _sub(B, n, rng)) for _ in range(repeats)], float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def noise_floor(observed, n, repeats=20, seed=0):
    """E-distance between two DISJOINT random halves of `observed`, each of size n, repeated.
    This is what a perfect prediction would score (sampling noise only)."""
    rng = np.random.default_rng(seed)
    obs = np.asarray(observed, float)
    vals = []
    for _ in range(repeats):
        idx = rng.permutation(len(obs))
        vals.append(_edist(obs[idx[:n]], obs[idx[n:2 * n]]))
    vals = np.array(vals, float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def calibrate_edistance(observed, reference, n=None, repeats=20, seed=0, z=2.0, max_n=300):
    """Calibrate one perturbation's propagation E-distance scale.

    observed  : observed perturbed-niche cells (the ground-truth distribution)
    reference : control/reference-niche cells (the 'no effect' anchor)

    Returns dict with floor (+std), S (+std), signal_gap, gap_z, has_signal, n.
    has_signal is True when S sits z standard deviations above the floor's noise band, i.e.
    there is a real niche shift to predict. Sample sizes are matched to n for all distances.
    """
    observed = np.asarray(observed, float)
    reference = np.asarray(reference, float)
    if n is None:
        n = min(len(observed) // 2, len(reference), max_n)
    n = max(2, int(n))
    floor_m, floor_s = noise_floor(observed, n, repeats, seed)
    S_m, S_s = edist_matched(observed, reference, n, repeats, seed + 1)
    gap = S_m - floor_m
    spread = float(np.sqrt(floor_s ** 2 + S_s ** 2)) or 1e-12
    return {
        "floor": floor_m, "floor_std": floor_s,
        "S": S_m, "S_std": S_s,
        "signal_gap": gap, "gap_z": gap / spread,
        "has_signal": bool(gap > z * spread),
        "n": n,
    }


def skill_score(model_error, cal):
    """Convert a model's propagation error into a 0..1 skill score using a calibration dict.

    skill = (S - model_error) / (S - floor). NaN when the perturbation has no signal (S ~ floor),
    because the denominator is then a tiny, ill-conditioned number. skill > 1 (model_error below
    the noise floor) is clipped to 1.0 and flagged via `clipped`.
    """
    if not cal["has_signal"]:
        return {"skill": float("nan"), "raw": float("nan"), "clipped": False, "has_signal": False}
    denom = cal["S"] - cal["floor"]
    raw = (cal["S"] - model_error) / denom
    return {"skill": float(min(raw, 1.0)), "raw": float(raw),
            "clipped": bool(raw > 1.0), "has_signal": True}


def propagation_skill(predicted, observed, reference, repeats=20, seed=0):
    """End-to-end: calibrate, compute the model's matched-n error, return the skill dict
    (with the model_error and the calibration attached)."""
    cal = calibrate_edistance(observed, reference, repeats=repeats, seed=seed)
    err, _ = edist_matched(np.asarray(predicted, float), np.asarray(observed, float),
                           cal["n"], repeats, seed + 2)
    out = skill_score(err, cal)
    out["model_error"] = err
    out["calibration"] = cal
    return out
