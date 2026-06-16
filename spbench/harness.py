import numpy as np
from .graph import neighbors_of
from .reference import match_reference_centers
from .propagation_gt import propagation_gt
from .metrics import get_metric

def _bystanders(data, center, edges):
    nb = neighbors_of(center, edges)
    return nb[~data.is_perturbed[nb]]

def fill_2x2(data, perturbation, edges, seed_model, baseline_prop, learned_prop, k_ref=5):
    """Fill the seed×propagation 2×2 for one perturbation.
    Rows = {GT seed, Model seed}, Cols = {baseline prop, learned prop}.
    Each cell scores propagation E-distance vs the observed perturbed-niche distribution."""
    energy = get_metric("energy")
    centers = np.where(data.perturbation == perturbation)[0]
    gt = propagation_gt(data, perturbation, edges, k_ref=k_ref)
    observed = gt["perturbed_niche"]

    refs = match_reference_centers(data, centers, k=k_ref)

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
            preds.append(prop_model.propagate(data.X, edges, c, seed_state, nb))
        return np.vstack(preds) if preds else np.zeros((0, data.n_genes))

    cells = {
        "1": collect(True, baseline_prop),
        "2": collect(True, learned_prop),
        "3": collect(False, baseline_prop),
        "4": collect(False, learned_prop),
    }
    return {k: {"energy_prop": energy.compute(v, observed)} for k, v in cells.items()}
