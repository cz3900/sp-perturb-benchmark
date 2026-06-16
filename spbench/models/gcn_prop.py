import numpy as np
import torch
import torch.nn as nn
from .base import PropModel
from . import register
from ..graph import _neighbor_index

def _norm_adj(edges, n):
    """Row-normalized NEIGHBOR adjacency (no self-loops) as a sparse torch tensor."""
    src, dst = edges
    idx = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    deg = np.bincount(src, minlength=n).clip(min=1)
    val = torch.tensor(1.0 / deg[src], dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()

class _Net(nn.Module):
    def __init__(self, g, h):
        super().__init__()
        self.l1 = nn.Linear(g, h)
        self.l2 = nn.Linear(h, g)

    def forward(self, x, A):
        # ONE graph aggregation step (no self-loops) then 2-layer MLP.
        # Applying A only once guarantees output[i] depends solely on x[neighbors(i)],
        # never on x[i] itself — this is the leakage guard.
        agg = torch.sparse.mm(A, x)               # aggregate neighbours (no self-loop)
        h = torch.relu(self.l1(agg))
        return self.l2(h)

@register
class SimpleGCN(PropModel):
    """2-layer GCN trained self-supervised: predict each cell from its NEIGHBOURS only
    (no self-loop => cannot copy itself). In-silico perturbation: set center=seed, run
    forward from the reference state, read the neighbours. This is a minimal SpatialProp."""
    name = "gcn_prop"

    def __init__(self, hidden=64, epochs=30, lr=1e-2, hops=2, seed=0):
        self.hidden, self.epochs, self.lr, self.hops, self.seed = hidden, epochs, lr, hops, seed

    def _adj(self, edges, n):
        """Row-normalized adjacency is identical for every epoch and every center, so build
        it once and cache it. Rebuilding + coalesce() per propagate() call was a hot spot."""
        key = (id(edges), n)
        if getattr(self, "_adj_key", None) != key:
            self._adj_cache, self._adj_key = _norm_adj(edges, n), key
        return self._adj_cache

    def _base_tensor(self, X_reference):
        """Cache the float32 reference state (constant across centers)."""
        key = id(X_reference)
        if getattr(self, "_xref_key", None) != key:
            self._xref_t = torch.as_tensor(np.asarray(X_reference, np.float32))
            self._xref_key = key
        return self._xref_t

    def _loss_once(self, train, edges):
        torch.manual_seed(self.seed)
        net = _Net(train.n_genes, self.hidden)
        A = self._adj(edges, train.n_cells)
        x = torch.tensor(train.X, dtype=torch.float32)
        with torch.no_grad():
            return float(((net(x, A) - x) ** 2).mean())

    def fit(self, train, edges):
        torch.manual_seed(self.seed)
        self.net_ = _Net(train.n_genes, self.hidden)
        A = self._adj(edges, train.n_cells)
        x = torch.tensor(train.X, dtype=torch.float32)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = ((self.net_(x, A) - x) ** 2).mean()
            loss.backward(); opt.step()
        return self

    def forward_numpy(self, X, edges):
        A = self._adj(edges, X.shape[0])
        with torch.no_grad():
            return self.net_(torch.tensor(X, dtype=torch.float32), A).numpy()

    def _propagate_full(self, X_reference, edges, center, seed_state, neighbors):
        """Reference implementation: forward over the WHOLE graph. Kept for validation."""
        A = self._adj(edges, np.asarray(X_reference).shape[0])
        seed_t = torch.as_tensor(np.asarray(seed_state, np.float32))
        xt = self._base_tensor(X_reference).clone()
        xt[center] = seed_t
        with torch.no_grad():
            for _ in range(self.hops):
                xt = self.net_(xt, A)
                xt[center] = seed_t
        return xt.numpy()[neighbors]

    def _local_subgraph(self, edges, read, center):
        """Nodes whose values can influence the readout at `read` after `hops` aggregations:
        read ∪ N(read) ∪ ... ∪ N^hops(read), plus the (pinned) center. Returns sorted global
        ids V and a global->local dict."""
        indptr, dst = _neighbor_index(edges)
        seen = set(int(r) for r in np.asarray(read).tolist())
        frontier = list(seen)
        for _ in range(self.hops):
            nxt = []
            for u in frontier:
                for w in dst[indptr[u]:indptr[u + 1]].tolist():
                    if w not in seen:
                        seen.add(w); nxt.append(w)
            frontier = nxt
        seen.add(int(center))
        V = np.array(sorted(seen), dtype=np.int64)
        loc = {int(v): i for i, v in enumerate(V.tolist())}
        return V, loc, indptr, dst

    def propagate(self, X_reference, edges, center, seed_state, neighbors):
        """Same result as `_propagate_full` (within fp tolerance) but forwards only on the local
        neighbourhood that the readout can depend on — ~100x cheaper on large graphs."""
        read = np.asarray(neighbors)
        ng = np.asarray(X_reference).shape[1]
        if read.size == 0:
            return np.zeros((0, ng), dtype=np.float32)
        V, loc, indptr, dst = self._local_subgraph(edges, read, center)
        nV = len(V)
        # local row-normalized adjacency, using GLOBAL out-degree for normalization
        rows, cols, vals = [], [], []
        for v in V.tolist():
            a, b = int(indptr[v]), int(indptr[v + 1])
            inv = 1.0 / max(1, b - a)
            lv = loc[v]
            for w in dst[a:b].tolist():
                lw = loc.get(w)
                if lw is not None:
                    rows.append(lv); cols.append(lw); vals.append(inv)
        if rows:
            idx = torch.tensor(np.array([rows, cols]), dtype=torch.long)
            A_local = torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32),
                                              (nV, nV)).coalesce()
        else:
            A_local = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long),
                                              torch.zeros(0, dtype=torch.float32), (nV, nV)).coalesce()
        seed_t = torch.as_tensor(np.asarray(seed_state, np.float32))
        lc = loc[int(center)]
        x = self._base_tensor(X_reference)[torch.as_tensor(V)].clone()   # (|V|, n_genes)
        x[lc] = seed_t
        with torch.no_grad():
            for _ in range(self.hops):
                x = self.net_(x, A_local)
                x[lc] = seed_t
        read_loc = np.array([loc[int(r)] for r in read.tolist()])
        return x.numpy()[read_loc]
