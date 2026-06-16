import numpy as np
import torch
import torch.nn as nn
from .base import PropModel
from . import register

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

    def _loss_once(self, train, edges):
        torch.manual_seed(self.seed)
        net = _Net(train.n_genes, self.hidden)
        A = _norm_adj(edges, train.n_cells)
        x = torch.tensor(train.X, dtype=torch.float32)
        with torch.no_grad():
            return float(((net(x, A) - x) ** 2).mean())

    def fit(self, train, edges):
        torch.manual_seed(self.seed)
        self.net_ = _Net(train.n_genes, self.hidden)
        A = _norm_adj(edges, train.n_cells)
        x = torch.tensor(train.X, dtype=torch.float32)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = ((self.net_(x, A) - x) ** 2).mean()
            loss.backward(); opt.step()
        return self

    def forward_numpy(self, X, edges):
        A = _norm_adj(edges, X.shape[0])
        with torch.no_grad():
            return self.net_(torch.tensor(X, dtype=torch.float32), A).numpy()

    def propagate(self, X_reference, edges, center, seed_state, neighbors):
        X = np.asarray(X_reference, float).copy()       # start from REFERENCE state (no leakage)
        X[center] = seed_state                          # inject only the perturbed center
        A = _norm_adj(edges, X.shape[0])
        xt = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(self.hops):
                xt = self.net_(xt, A)
                xt[center] = torch.tensor(seed_state, dtype=torch.float32)   # pin center
        return xt.numpy()[neighbors]
