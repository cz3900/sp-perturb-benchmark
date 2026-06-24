import numpy as np
from .propagation_gt import propagation_gt, _bystander_neighbors
from .metrics.energy import energy_distance


def _matched_energy(A, B, rng, max_n):
    A = np.asarray(A, float); B = np.asarray(B, float)
    if len(A) < 1 or len(B) < 1:
        return float("nan")
    n = max(2, min(len(A), len(B), max_n))
    a = A[rng.choice(len(A), n, replace=False)] if len(A) > n else A
    b = B[rng.choice(len(B), n, replace=False)] if len(B) > n else B
    return energy_distance(a, b)


def permutation_null(data, perturbation, edges, n_perm=50, seed=0, max_n=300):
    """Empirical null for a perturbation's niche shift (see Plan 3)."""
    gt = propagation_gt(data, perturbation, edges)
    perturbed, reference, centers = gt["perturbed_niche"], gt["reference_niche"], gt["centers"]
    if len(perturbed) == 0 or len(reference) == 0 or len(centers) == 0:
        return {"null": [], "real": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(seed)
    real = _matched_energy(perturbed, reference, rng, max_n)
    pool = np.where(~data.is_perturbed)[0]
    null = []
    k = min(len(centers), len(pool))
    for _ in range(n_perm):
        fake = rng.choice(pool, k, replace=False)
        nb = [_bystander_neighbors(data, c, edges) for c in fake]
        nb = np.concatenate(nb) if any(len(x) for x in nb) else np.array([], int)
        if len(nb) == 0:
            continue
        null.append(_matched_energy(data.X[nb], reference, rng, max_n))
    null = [x for x in null if np.isfinite(x)]
    p = (np.sum(np.asarray(null) >= real) + 1) / (len(null) + 1) if null else float("nan")
    return {"null": null, "real": float(real), "p": float(p)}
