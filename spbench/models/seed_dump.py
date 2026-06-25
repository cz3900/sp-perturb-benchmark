"""Generic loader for any offline seed model that dumps `{P}_seed.h5ad` (/X aligned to centers
order + /obs/center_idx). scGEN / CPA / GEARS / biolord all train in their own conda env and share
this dump contract, so they need no per-model loader — just SeedDumpModel(name, prediction_paths).
ScgenSeedModel is now a thin subclass of this."""
import numpy as np, h5py
from .base import SeedModel
from .concert_model import read_h5ad_X


def read_center_idx(path):
    """Read /obs/center_idx — the StandardData center indices each dumped row aligns to."""
    with h5py.File(path, "r") as f:
        return np.asarray(f["obs"]["center_idx"], dtype=np.int64)


class SeedDumpModel(SeedModel):
    """prediction_paths : {perturbation: path to its `{P}_seed.h5ad`}."""
    def __init__(self, name, prediction_paths):
        self.name = name
        self.prediction_paths = dict(prediction_paths)
        self._cache = {}

    def fit(self, train):                       # trained offline in its own env; nothing to do
        return self

    def _load(self, perturbation):
        if perturbation not in self._cache:
            self._cache[perturbation] = np.asarray(read_h5ad_X(self.prediction_paths[perturbation]),
                                                   float)
        return self._cache[perturbation]

    def centers(self, perturbation):
        return read_center_idx(self.prediction_paths[perturbation])

    def predict_seed(self, perturbation, reference_cells):
        """The cached (n_centers, G) array aligned to centers order. reference_cells is part of the
        SeedModel ABC but the prediction is the offline-aligned array, so it is unused."""
        return self._load(perturbation)
