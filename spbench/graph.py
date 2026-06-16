import numpy as np
from sklearn.neighbors import NearestNeighbors

def build_knn_graph(data, k: int = 15) -> np.ndarray:
    """Directed kNN edges built WITHIN each batch (slice). Returns (2, n_edges) int array
    of [src, dst] where dst is a neighbor of src. No self-loops, no cross-batch edges."""
    src_all, dst_all = [], []
    for b in np.unique(data.batch):
        idx = np.where(data.batch == b)[0]
        if len(idx) <= k:
            continue
        nn = NearestNeighbors(n_neighbors=k + 1).fit(data.coords[idx])
        _, nbr = nn.kneighbors(data.coords[idx])
        for local_i in range(len(idx)):
            for local_j in nbr[local_i, 1:]:        # skip self (col 0)
                src_all.append(idx[local_i])
                dst_all.append(idx[local_j])
    return np.array([src_all, dst_all], dtype=int)

def neighbors_of(node: int, edges: np.ndarray) -> np.ndarray:
    return edges[1][edges[0] == node]
