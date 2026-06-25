import numpy as np
from .graph import neighbors_of
from .reference_aggregate import control_reference_centers

def _bystander_neighbors(data, center, edges):
    """Neighbors of `center` that are NOT themselves perturbed (the bystanders)."""
    nb = neighbors_of(center, edges)
    return nb[~data.is_perturbed[nb]]

def propagation_gt(data, perturbation: str, edges: np.ndarray, k_ref: int = 5) -> dict:
    """Observed propagation effect for `perturbation`: the bystander neighbours of all perturbed
    centers vs the bystander neighbours of the SAME-CELL-TYPE control cells — a sample-level
    aggregate-control reference, NOT expression-matched reference centers. So `reference_niche` is
    the 'no-effect' niche (e_null in spbench.compare): the neighbourhood of the average unperturbed
    cell of that type. `k_ref` is kept for signature compatibility but no longer selects centers."""
    centers = np.where(data.perturbation == perturbation)[0]
    pert_nb = np.concatenate([_bystander_neighbors(data, c, edges) for c in centers]) \
        if len(centers) else np.array([], int)
    refs = control_reference_centers(data, centers)
    ref_centers = np.unique(np.concatenate(refs)) if len(refs) else np.array([], int)
    ref_nb = np.concatenate([_bystander_neighbors(data, c, edges) for c in ref_centers]) \
        if len(ref_centers) else np.array([], int)
    return {
        "perturbed_niche": data.X[pert_nb],
        "reference_niche": data.X[ref_nb],
        "centers": centers, "ref_centers": ref_centers,
        "pert_nb": pert_nb, "ref_nb": ref_nb,   # bystander indices (so a caller can re-index into another space)
    }
