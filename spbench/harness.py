import numpy as np
from .graph import neighbors_of
from .reference import match_reference_centers
from .propagation_gt import propagation_gt
from .metrics import get_metric

def _bystanders(data, center, edges):
    nb = neighbors_of(center, edges)
    return nb[~data.is_perturbed[nb]]

def _control_reference(data):
    """Reference ('unperturbed') state matrix: each cell -> mean expression of CONTROL cells of
    its cell type (global control mean as fallback). Propagation starts from this instead of the
    observed matrix, so a prediction can never trivially reproduce the observed perturbed niche
    (which would be leakage). With a control reference the seed shift = perturbed - control is
    real, and the predicted niche is genuinely compared against the observed perturbed niche."""
    ctrl = data.is_control
    global_mean = data.X[ctrl].mean(0) if ctrl.any() else data.X.mean(0)
    X_ref = np.tile(global_mean, (data.n_cells, 1)).astype(float)
    for ct in np.unique(data.cell_type):
        m = ctrl & (data.cell_type == ct)
        if m.any():
            X_ref[data.cell_type == ct] = data.X[m].mean(0)
    return X_ref

def _control_residuals(data):
    """Per-cell-type pool of CONTROL residuals (X_control - its cell-type control mean), plus a
    global pool fallback under key None.

    A propagation model emits one vector per cell — the conditional *mean*. The observed niche is
    a full-variance cloud (spread ~6), so scoring a near-degenerate mean-field cloud with the
    energy distance (a *distributional* metric) inflates it structurally, regardless of whether
    the predicted shift is right. Adding a sampled control residual to each predicted cell is the
    deterministic analogue of a generative model drawing per-cell samples around its mean: it
    restores realistic per-cell biological variance without moving the mean, so the energy
    distance measures the predicted SHIFT fairly. Residuals come only from CONTROL cells, never
    from the observed perturbed niche, so they cannot leak."""
    ctrl = data.is_control
    pools = {}
    gmean = data.X[ctrl].mean(0) if ctrl.any() else data.X.mean(0)
    pools[None] = (data.X[ctrl] - gmean) if ctrl.any() else (data.X - gmean)
    for ct in np.unique(data.cell_type):
        m = ctrl & (data.cell_type == ct)
        if m.any():
            pools[ct] = data.X[m] - data.X[m].mean(0)
    return pools

def _draw_residuals(pools, cell_types, rng):
    """One sampled residual per cell, drawn from its cell-type pool (global pool as fallback)."""
    g = pools[None]
    out = np.empty((len(cell_types), g.shape[1]), float)
    for i, ct in enumerate(cell_types):
        R = pools.get(ct, g)
        if len(R) == 0:
            R = g
        out[i] = R[rng.integers(len(R))] if len(R) else 0.0
    return out

def fill_2x2(data, perturbation, edges, seed_model, baseline_prop, learned_prop, k_ref=5,
             X_ref=None, return_niches=False, residuals=None, noise_seed=0):
    """Fill the seed×propagation 2×2 for one perturbation.
    Rows = {GT seed, Model seed}, Cols = {baseline prop, learned prop}.
    Each cell scores propagation E-distance vs the observed perturbed-niche distribution.

    `residuals` (from `_control_residuals`) gives every predicted cell realistic per-cell
    variance so the energy distance compares the predicted *shift* fairly instead of penalising
    the variance collapse of a mean-field prediction (see `_control_residuals`). The residual
    draws are reseeded identically for each of the four cells, so e1..e4 differ only by their
    mean field. Pass residuals=None to score the raw mean-only predictions.

    With return_niches=True the grid also carries `_niches` = {observed, reference, "3", "4"}
    (the predicted bystander-niche arrays for the two deployable models) so skill scores can be
    computed downstream via spbench.calibrate."""
    energy = get_metric("energy")
    centers = np.where(data.perturbation == perturbation)[0]
    gt = propagation_gt(data, perturbation, edges, k_ref=k_ref)
    observed = gt["perturbed_niche"]

    refs = match_reference_centers(data, centers, k=k_ref)
    if X_ref is None:
        X_ref = _control_reference(data)   # propagation starts from the control niche, NOT the observed one

    def collect(use_gt_seed, prop_model):
        rng = np.random.default_rng(noise_seed)   # identical residual draws across the 4 cells
        preds = []
        for c, rc in zip(centers, refs):
            nb = _bystanders(data, c, edges)
            if len(nb) == 0:
                continue
            if use_gt_seed:
                seed_state = data.X[c]                                     # oracle: true perturbed center
            else:
                # model seed predicts from MATCHED CONTROL cells, never the center's own value
                seed_state = seed_model.predict_seed(perturbation, data.X[rc]).mean(0)
            pred = prop_model.propagate(X_ref, edges, c, seed_state, nb)
            if residuals is not None:                                      # distributional readout
                pred = pred + _draw_residuals(residuals, data.cell_type[nb], rng)
            preds.append(pred)
        return np.vstack(preds) if preds else np.zeros((0, data.n_genes))

    cells = {
        "1": collect(True, baseline_prop),
        "2": collect(True, learned_prop),
        "3": collect(False, baseline_prop),
        "4": collect(False, learned_prop),
    }
    grid = {k: {"energy_prop": energy.compute(v, observed)} for k, v in cells.items()}
    if return_niches:
        grid["_niches"] = {"observed": observed, "reference": gt["reference_niche"],
                           "3": cells["3"], "4": cells["4"]}
    return grid
