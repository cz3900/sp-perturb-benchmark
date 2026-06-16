import numpy as np
from sklearn.neighbors import NearestNeighbors


def match_reference_centers(data, centers: np.ndarray, k: int = 5) -> list:
    """For each perturbed center, return k control cells matched by cell type + expression
    nearest-neighbour (feature-space match, NOT physical pairing)."""
    out = []
    ctrl_idx = np.where(data.is_control)[0]
    for c in centers:
        same = ctrl_idx[data.cell_type[ctrl_idx] == data.cell_type[c]]
        if len(same) == 0:
            same = ctrl_idx
        nn = NearestNeighbors(n_neighbors=min(k, len(same))).fit(data.X[same])
        _, j = nn.kneighbors(data.X[c:c + 1])
        out.append(same[j[0]])
    return out
