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

def fill_2x2(data, perturbation, edges, seed_model, baseline_prop, learned_prop, k_ref=5,
             X_ref=None, return_niches=False):
    """Fill the seed×propagation 2×2 for one perturbation.
    Rows = {GT seed, Model seed}, Cols = {baseline prop, learned prop}.
    Each cell scores propagation E-distance vs the observed perturbed-niche distribution.
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
            preds.append(prop_model.propagate(X_ref, edges, c, seed_state, nb))
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
