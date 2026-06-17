import numpy as np
from scipy.spatial.distance import cdist
from .base import Metric
from . import register

def energy_distance(X, Y) -> float:
    """Raw energy distance 2*mean||X-Y|| - mean||X-X'|| - mean||Y-Y'|| with NO internal
    subsampling. The canonical primitive: callers that control their own sample size (e.g. the
    matched-n baseline comparison in spbench.compare) use this directly; the EnergyDistance metric
    subsamples first, then calls it. NaN for empty inputs, clamped at 0."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return float("nan")
    return float(max(0.0, 2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean()))

class EnergyDistance(Metric):
    """E^2(X,Y) = 2*mean||X-Y|| - mean||X-X'|| - mean||Y-Y'||.
    pred=X (predicted cell group), gt=Y (observed group). Subsample for speed, then call the
    shared `energy_distance` primitive."""
    name = "energy"
    higher_is_better = False

    def __init__(self, max_n: int = 500, seed: int = 0):
        self.max_n = max_n
        self.rng = np.random.default_rng(seed)

    def _sub(self, A):
        if A.shape[0] > self.max_n:
            idx = self.rng.choice(A.shape[0], self.max_n, replace=False)
            return A[idx]
        return A

    def compute(self, pred, gt, context=None) -> float:
        return energy_distance(self._sub(np.asarray(pred, float)), self._sub(np.asarray(gt, float)))

register(EnergyDistance())
