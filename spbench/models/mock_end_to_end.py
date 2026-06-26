import numpy as np
from .base import EndToEndModel
from ..graph import neighbors_of


class MockEndToEnd(EndToEndModel):
    """A stand-in end-to-end model for testing the external-model path: its predicted bystander
    niche is the OBSERVED bystander niche plus gaussian noise (so its niche PCC-delta is near 1,
    clearly above the null). NOT a real model — it reads observed expression, which a real model may
    not. Use only to exercise plumbing/plots."""
    name = "mock_end_to_end"

    def __init__(self, noise=0.3, seed=0):
        self.noise = float(noise); self.seed = int(seed)

    def fit(self, train, edges):
        return self

    def _bystanders(self, data, perturbation, edges):
        centers = np.where(data.perturbation == perturbation)[0]
        nbs = [neighbors_of(c, edges) for c in centers]
        nbs = [nb[~data.is_perturbed[nb]] for nb in nbs]
        return np.concatenate(nbs) if any(len(x) for x in nbs) else np.array([], int)

    def predict_niche(self, data, perturbation, edges):
        idx = self._bystanders(data, perturbation, edges)
        if len(idx) == 0:
            return np.zeros((0, data.n_genes), float)
        rng = np.random.default_rng(self.seed)
        return data.X[idx] + rng.normal(scale=self.noise, size=(len(idx), data.n_genes))

    def predict(self, perturbation, reference_cells, edges, center, neighbors):
        raise NotImplementedError("MockEndToEnd only implements predict_niche (niche scoring path).")
