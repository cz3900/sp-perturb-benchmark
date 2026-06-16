import numpy as np
from .graph import neighbors_of
from .reference import match_reference_centers

def _bystander_neighbors(data, center, edges):
    """Neighbors of `center` that are NOT themselves perturbed (the bystanders)."""
    nb = neighbors_of(center, edges)
    return nb[~data.is_perturbed[nb]]

def propagation_gt(data, perturbation: str, edges: np.ndarray, k_ref: int = 5) -> dict:
    """Observed propagation effect for `perturbation`: pool the bystander neighbours of all
    perturbed centers vs the bystander neighbours of matched control centers."""
    centers = np.where(data.perturbation == perturbation)[0]
    pert_nb = np.concatenate([_bystander_neighbors(data, c, edges) for c in centers]) \
        if len(centers) else np.array([], int)
    refs = match_reference_centers(data, centers, k=k_ref)
    ref_centers = np.unique(np.concatenate(refs)) if len(refs) else np.array([], int)
    ref_nb = np.concatenate([_bystander_neighbors(data, c, edges) for c in ref_centers]) \
        if len(ref_centers) else np.array([], int)
    return {
        "perturbed_niche": data.X[pert_nb],
        "reference_niche": data.X[ref_nb],
        "centers": centers, "ref_centers": ref_centers,
    }
