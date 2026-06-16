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

_NBR_CACHE = {}


def _neighbor_index(edges: np.ndarray):
    """CSR-style src->dst index, built once per edges array (O(degree) lookups instead of
    scanning all edges per node). Stable sort preserves original neighbour order."""
    cached = _NBR_CACHE.get(id(edges))
    if cached is not None and cached[0] is edges:
        return cached[1], cached[2]
    if edges.shape[1] == 0:
        indptr, dst = np.zeros(1, dtype=int), edges[1]
    else:
        order = np.argsort(edges[0], kind="stable")
        src_sorted, dst = edges[0][order], edges[1][order]
        n = int(edges[0].max()) + 1
        indptr = np.searchsorted(src_sorted, np.arange(n + 1))
    _NBR_CACHE[id(edges)] = (edges, indptr, dst)
    return indptr, dst


def neighbors_of(node: int, edges: np.ndarray) -> np.ndarray:
    indptr, dst = _neighbor_index(edges)
    if node + 1 >= len(indptr):
        return dst[len(dst):]
    return dst[indptr[node]:indptr[node + 1]]
