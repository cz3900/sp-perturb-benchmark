import numpy as np
from .propagation_gt import propagation_gt
from .compare import compare_to_baseline


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
