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


def _categories(data) -> np.ndarray:
    """Global sorted cell-type vocabulary (stable column order for the (C,) composition)."""
    return np.array(sorted(np.unique(data.cell_type).tolist()), dtype=object)


def n_hop_neighbors(node: int, edges: np.ndarray, hops: int = 1) -> np.ndarray:
    """All cells reachable within `hops` steps of `node`, excluding the node itself.

    hops=1 -> direct neighbors (same set as graph.neighbors_of). Larger hops widen the
    niche scale (configurable, used by the sensitivity sweep). Pure BFS over `edges`,
    so it works for any (2, n_edges) graph regardless of how it was built.
    """
    frontier = {int(node)}
    seen = {int(node)}
    for _ in range(max(1, int(hops))):
        nxt = set()
        for u in frontier:
            for v in neighbors_of(u, edges):
                iv = int(v)
                if iv not in seen:
                    nxt.add(iv)
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    seen.discard(int(node))
    return np.array(sorted(seen), dtype=int)


def compute_niche_composition(data, edges: np.ndarray, hops: int = 1) -> np.ndarray:
    """Per-cell neighborhood cell-type composition -> (n_cells, C) row-simplex matrix.

    For each cell, count the cell types of its `hops`-neighborhood and L1-normalize.
    C = number of global cell types (column order = _categories(data)). Cells with no
    neighbors get an all-zero row (the aggregate control handles them downstream).

    `edges` may come from build_spatial_graph (squidpy) or graph.build_knn_graph — this
    function is pure numpy and carries no squidpy runtime dependency. `hops` makes the
    niche scale configurable (default 1 = direct neighbors).
    """
    cats = _categories(data)
    code = {ct: i for i, ct in enumerate(cats)}
    ct_code = np.array([code[c] for c in data.cell_type], dtype=int)
    C = len(cats)
    comp = np.zeros((data.n_cells, C), dtype=float)
    for i in range(data.n_cells):
        nb = neighbors_of(i, edges) if hops == 1 else n_hop_neighbors(i, edges, hops=hops)
        if len(nb) == 0:
            continue
        counts = np.bincount(ct_code[nb], minlength=C).astype(float)
        comp[i] = counts / counts.sum()
    return comp
