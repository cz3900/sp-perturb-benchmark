"""Niche definition: squidpy spatial graph + neighborhood cell-type composition.

The graph is built WITHIN each batch (slice) — squidpy has no batch concept, so we
run gr.spatial_neighbors per batch and remap to global cell indices. Output matches
spbench.graph.build_knn_graph exactly: a (2, n_edges) int array [src, dst], no self
loops, no cross-batch edges, so it is a drop-in for neighbors_of / harness._bystanders.

squidpy/anndata are imported lazily inside build_spatial_graph so that importing this
module (and using the pure-numpy compute_niche_composition, added in G2.2) carries no
squidpy runtime dependency. graph.neighbors_of is imported at module level — graph.py
depends only on numpy/sklearn.
"""
import numpy as np

from .graph import neighbors_of  # noqa: F401  (used by compute_niche_composition in G2.2)


def build_spatial_graph(data, n_neighs: int = 15, coord_type: str = "generic",
                        radius: float | None = None) -> np.ndarray:
    """squidpy-backed spatial graph, built per batch. Returns (2, n_edges) int [src, dst].

    n_neighs : kNN degree (default scale, matches build_knn_graph's k spirit).
    coord_type : passed to squidpy ('generic' for arbitrary coords; 'grid' for lattices).
    radius : if given, squidpy uses a distance threshold instead of fixed kNN.
    """
    import anndata as ad
    import squidpy as sq

    src_all, dst_all = [], []
    for b in np.unique(data.batch):
        idx = np.where(data.batch == b)[0]
        if len(idx) <= 1:
            continue
        a = ad.AnnData(X=np.zeros((len(idx), 1), dtype=float))
        a.obsm["spatial"] = np.asarray(data.coords[idx], dtype=float)
        kwargs = dict(coord_type=coord_type)
        if radius is not None:
            kwargs["radius"] = float(radius)
        else:
            kwargs["n_neighs"] = int(n_neighs)
        sq.gr.spatial_neighbors(a, **kwargs)
        conn = a.obsp["spatial_connectivities"].tocoo()
        for li, lj in zip(conn.row, conn.col):
            if li == lj:                      # drop any self loop
                continue
            src_all.append(int(idx[li]))      # map local -> global
            dst_all.append(int(idx[lj]))
    if not src_all:
        return np.zeros((2, 0), dtype=int)
    return np.array([src_all, dst_all], dtype=int)
