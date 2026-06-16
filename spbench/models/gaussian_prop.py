import numpy as np
from .base import PropModel
from . import register

@register
class GaussianProp(PropModel):
    """Baseline: neighbour = reference state + distance-weighted (Gaussian) share of the
    center's seed shift. No learning."""
    name = "gaussian_prop"

    def __init__(self, bandwidth: float = 2.0):
        self.bandwidth = bandwidth

    def fit(self, train, edges):
        self.coords_ = train.coords
        return self

    def propagate(self, X_reference, edges, center, seed_state, neighbors):
        ref = np.asarray(X_reference, float)
        shift = seed_state - ref[center]
        d = np.linalg.norm(self.coords_[neighbors] - self.coords_[center], axis=1)
        w = np.exp(-(d ** 2) / (2 * self.bandwidth ** 2))
        return ref[neighbors] + w[:, None] * shift[None, :]
