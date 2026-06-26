import numpy as np
from .propagation_gt import propagation_gt
from .compare import compare_to_baseline, evaluate_seed
from .reference_aggregate import control_reference_centers


def row_normalize(X, target_sum=1e4):
    """Per-cell (per-row) normalize_total: each row scaled to sum to `target_sum`, pure numpy
    (no scanpy). Rows whose sum is 0 are returned as all-zero (NOT nan).

    End-to-end model dumps live in their own native spaces (SpatialProp = normalize_total, CONCERT
    = raw counts); the caller row-normalizes both data.X and each model's prediction to a common
    target before scoring. PCC-delta is immune to any residual global scale, so this is a pure
    linear alignment of per-cell library size."""
    X = np.asarray(X, float)
    s = X.sum(1, keepdims=True)
    out = np.zeros_like(X)
    nz = s[:, 0] > 0
    out[nz] = X[nz] / s[nz] * target_sum
    return out


def score_external_seed(data, perturbation, pred_full, eval_X=None, repeats=20, seed=0, max_n=300):
    """Score an end-to-end model's prediction for the PERTURBED CENTER cells themselves (seed),
    symmetric to `score_external_niche` (which scores the bystander niche). Returns the
    `evaluate_seed` dict (pcc_delta / mse / mag / n / e_samples).

    pred_full : (n_cells, n_genes) the model's predicted expression for ALL cells, already in the
                same space as data.X (the caller row-normalizes both to a common target first).
    seed_obs  : the observed expression of the perturbed centers (data.X[centers]).
    seed_pred : the model's prediction sliced to those same centers (pred_full[centers]).
    seed_ref  : the matched control reference for those centers — the SAME caliber as fill_2x2's
                seed_ref: the union of all same-cell-type control cells of the perturbed centers
                (control_reference_centers, observed in data.X). This is the shift baseline that
                pcc_delta's delta is taken against.

    Empty centers -> {pcc_delta, mse, mag = nan, n = 0}."""
    centers = np.where(data.perturbation == perturbation)[0]
    if len(centers) == 0:
        return {"pcc_delta": float("nan"), "mse": float("nan"), "mag": float("nan"), "n": 0}
    pred_full = np.asarray(pred_full, float)
    refs = control_reference_centers(data, centers)            # same caliber as harness.fill_2x2
    seed_ref_idx = np.unique(np.concatenate(refs)) if len(refs) else np.array([], int)
    niches = {"seed_obs": data.X[centers],
              "seed_pred": pred_full[centers],
              "seed_ref": data.X[seed_ref_idx]}
    return evaluate_seed(niches, eval_X=eval_X, repeats=repeats, seed=seed, max_n=max_n)


def score_external_niche(data, perturbation, edges, niche_pred, name="external",
                         residuals=None, eval_X=None, k_ref=5, repeats=20, seed=0, max_n=300):
    """Score an external/end-to-end model's predicted bystander niche on the same matched-n energy
    / gain / PCC-delta footing as the 2x2 cells. Builds the observed + no-effect-reference niches
    via propagation_gt and passes `niche_pred` through compare_to_baseline(extra={name: niche_pred}).
    Returns the compare_to_baseline dict (so res['e'][name], res['e_samples'][name], etc.).

    NOTE: only 'observed'/'reference' are supplied (no 2x2 cells) — compare_to_baseline builds its
    cell clouds via `for k in METHODS if k in niches`, so it tolerates their absence and scores just
    null + the external method here."""
    gt = propagation_gt(data, perturbation, edges, k_ref=k_ref)
    niches = {"observed": gt["perturbed_niche"], "reference": gt["reference_niche"]}
    return compare_to_baseline(niches, residuals=residuals, repeats=repeats, seed=seed,
                               max_n=max_n, extra={name: np.asarray(niche_pred, float)}, eval_X=eval_X)
