import numpy as np
from .propagation_gt import _bystander_neighbors
from .metrics.energy import energy_distance


def _matched_energy(A, B, rng, max_n):
    A = np.asarray(A, float); B = np.asarray(B, float)
    if len(A) < 1 or len(B) < 1:
        return float("nan")
    n = max(2, min(len(A), len(B), max_n))
    a = A[rng.choice(len(A), n, replace=False)] if len(A) > n else A
    b = B[rng.choice(len(B), n, replace=False)] if len(B) > n else B
    return energy_distance(a, b)


def _niche_cells(data, centers, edges):
    """Deduplicated bystander (non-perturbed) neighbour cells of `centers`. Dedup matters: a cell
    that neighbours several centers would otherwise appear once per center, and the duplication
    multiplicity differs between the perturbed centers and a random center set — a geometry artifact
    that biases the energy distance even with no biological effect."""
    nb = [_bystander_neighbors(data, c, edges) for c in centers]
    nb = np.concatenate(nb) if any(len(x) for x in nb) else np.array([], int)
    return np.unique(nb)


def permutation_null(data, perturbation, edges, n_perm=50, seed=0, max_n=300):
    """Empirical null for a perturbation's niche shift, as a two-sample RELABELING test.

    Statistic for a center set C: matched-n energy distance between C's deduplicated bystander-cell
    expression and a FIXED shared background = the expression of all non-perturbed cells.
        real    = statistic(perturbed centers)
        null_i  = statistic(|centers| random non-perturbed 'fake' centers)        (n_perm draws)
        p       = (#{null >= real} + 1) / (len(null) + 1)

    Why this construction (vs comparing to the control niche): under H0 (no niche effect) the
    perturbed centers are exchangeable with random non-perturbed centers, and BOTH are scored
    against the SAME background, so the null is a faithful relabeling distribution and an inert
    perturbation gives a ~uniform p (high, not falsely low). Comparing fake-control bystanders to a
    control-specific reference (an earlier design) was a near self-comparison (null collapses to ~0)
    while the perturbed bystanders are a different spatial subset, which pushed even inert
    perturbations to low p. A real niche shift makes the perturbed bystanders differ from the
    background -> real >> null -> low p. (Self-contamination of the background by the test set's own
    shifted bystanders is symmetric for real/null and only makes the planted case slightly
    conservative.)"""
    centers = np.where(data.perturbation == perturbation)[0]
    pert_cells = _niche_cells(data, centers, edges)
    bg = np.where(~data.is_perturbed)[0]
    if len(centers) == 0 or len(pert_cells) == 0 or len(bg) == 0:
        return {"null": [], "real": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(seed)
    Xbg = data.X[bg]
    real = _matched_energy(data.X[pert_cells], Xbg, rng, max_n)
    null, k = [], min(len(centers), len(bg))
    for _ in range(n_perm):
        fake = rng.choice(bg, k, replace=False)
        fake_cells = _niche_cells(data, fake, edges)
        if len(fake_cells) == 0:
            continue
        null.append(_matched_energy(data.X[fake_cells], Xbg, rng, max_n))
    null = [x for x in null if np.isfinite(x)]
    p = (np.sum(np.asarray(null) >= real) + 1) / (len(null) + 1) if null else float("nan")
    return {"null": null, "real": float(real), "p": float(p)}
