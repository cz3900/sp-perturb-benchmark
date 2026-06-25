import numpy as np
from sklearn.neighbors import NearestNeighbors
from .base import Metric
from . import register

def morans_i(values: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    """Global Moran's I of `values` over a kNN spatial graph (binary weights)."""
    values = np.asarray(values, float)
    n = len(values)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]                       # drop self
    z = values - values.mean()
    W = n * k
    num = sum(z[i] * z[idx[i]].sum() for i in range(n))
    den = (z ** 2).sum()
    if den == 0:
        return 0.0
    return float((n / W) * (num / den))

class MoranI(Metric):
    """Mean absolute gap in per-gene Moran's I between predicted and observed fields.
    pred/gt are (n_cells, n_genes); context must carry 'coords'. Lower = better."""
    name = "moran_gap"
    higher_is_better = False
    status = "planned"

    def __init__(self, k: int = 6, max_genes: int = 50):
        self.k = k
        self.max_genes = max_genes

    def compute(self, pred, gt, context=None) -> float:
        coords = context["coords"]
        pred, gt = np.asarray(pred, float), np.asarray(gt, float)
        g = min(self.max_genes, pred.shape[1])
        gaps = [abs(morans_i(pred[:, j], coords, self.k) - morans_i(gt[:, j], coords, self.k))
                for j in range(g)]
        return float(np.mean(gaps))

register(MoranI())
