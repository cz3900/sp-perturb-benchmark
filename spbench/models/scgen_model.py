"""scGEN as an offline, learned D1 SeedModel — a LOADER, mirroring ConcertModel.

scGEN trains in its own conda env (scvi-tools/jax/lightning pins conflict with the shared 3.11
venv), so it is run OFFLINE (scripts/scgen/run_scgen.py): export StandardData -> log-norm AnnData,
train one scGEN per perturbation, predict the perturbed state for the harness's per-center matched
controls, and dump the FINAL seed_pred array (n_centers x G), already aligned to centers order, to
`{P}_seed.h5ad`. This wrapper reads that array and serves it directly as the model seed — exactly
like ConcertModel serves a cached counterfactual.

Per the design's Part C fixes: predict_seed keeps the ABC's two-arg signature
(perturbation, reference_cells) and does NOT add a ref_idx third arg; the cached array IS the
prediction, so reference_cells is not used to re-derive it. The runner has already honored the
aggregate-control contract (it decodes the latent delta on the cell-type-mean control profile).
Constructed directly with prediction_paths (like ConcertModel) — NOT @register'd / auto-imported,
because __init__ requires prediction_paths.

Note: fit(train) is single-arg (SeedModel.fit, base.py:8), unlike ConcertModel.fit(train, edges).
"""
import numpy as np
import h5py
from .base import SeedModel
from .concert_model import read_h5ad_X


def read_center_idx(path):
    """Read /obs/center_idx (the StandardData center indices each seed_pred row aligns to)."""
    with h5py.File(path, "r") as f:
        return np.asarray(f["obs"]["center_idx"], dtype=np.int64)


class ScgenSeedModel(SeedModel):
    """Loader wrapper around scGEN's offline seed predictions.

    prediction_paths : {perturbation: path to its `{P}_seed.h5ad` (/X aligned to centers order)}.
    """
    name = "scgen"

    def __init__(self, prediction_paths):
        self.prediction_paths = dict(prediction_paths)
        self._cache = {}

    def fit(self, train) -> "ScgenSeedModel":      # trained offline in the scgen env; nothing to do
        return self

    def _load(self, perturbation):
        if perturbation not in self._cache:
            self._cache[perturbation] = np.asarray(read_h5ad_X(self.prediction_paths[perturbation]),
                                                   float)
        return self._cache[perturbation]

    def centers(self, perturbation):
        """The StandardData center indices the cached seed_pred rows align to (for fill_2x2 to
        index the same-space seed_obs / seed_ref)."""
        return read_center_idx(self.prediction_paths[perturbation])

    def predict_seed(self, perturbation, reference_cells) -> np.ndarray:
        """scGEN's offline-predicted perturbed-center expression for `perturbation`, as the cached
        (n_centers, n_genes) array already aligned to centers order. reference_cells is part of the
        SeedModel ABC contract but the prediction is the offline-aligned array, so it is unused."""
        return self._load(perturbation)
