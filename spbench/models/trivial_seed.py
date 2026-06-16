import numpy as np
from .base import SeedModel
from . import register

@register
class TrivialSeed(SeedModel):
    """Predicts reference + global mean perturbation shift (averaged over ALL seen
    perturbations). No gene features -> works identically for seen and unseen.
    This is the mandatory floor baseline (Ahlmann-Eltze)."""
    name = "trivial_seed"

    def fit(self, train):
        ctrl_mean = train.X[train.is_control].mean(0)
        shifts = []
        for p in train.perturbations():
            shifts.append(train.X[train.perturbation == p].mean(0) - ctrl_mean)
        self.global_shift_ = np.mean(shifts, 0) if shifts else np.zeros(train.n_genes)
        return self

    def predict_seed(self, perturbation, reference_cells):
        return np.asarray(reference_cells, float) + self.global_shift_
