"""SpatialProp as a black-box, offline end-to-end model in our benchmark (mirrors ConcertModel).

SpatialProp trains + predicts in its own torch/PyG env (incompatible with spbench's env), so we do
NOT import it here. Instead SpatialProp is run offline, producing a counterfactual `.h5ad` whose
predicted expression lives in `layers['predicted_tempered']` (NOT `/X`). This wrapper is a LOADER: it
reads that layer and extracts the predicted *bystander niche* (neighbours of the perturbed centers),
which scores in spbench.compare exactly like our baselines' predicted niches — slotting straight into
the same external-niche path as ConcertModel (run_benchmark external_models / compare_to_baseline
extra=). One `.h5ad` per evaluated perturbation.
"""
import numpy as np
import h5py
from .base import EndToEndModel
from ..graph import neighbors_of


def read_h5ad_layer(path, layer):
    """Read a dense (N, G) matrix from `/layers/<layer>` of an `.h5ad` via h5py — no anndata dep."""
    with h5py.File(path, "r") as f:
        return np.asarray(f["layers"][layer], float)


class SpatialPropModel(EndToEndModel):
    """Loader wrapper around SpatialProp's offline counterfactual predictions.

    prediction_paths : {perturbation: path to its SpatialProp counterfactual `.h5ad`}.
    layer            : the `.h5ad` layer holding the predicted expression (default tempered).
    """
    name = "spatialprop"

    def __init__(self, prediction_paths, layer="predicted_tempered"):
        self.prediction_paths = dict(prediction_paths)
        self.layer = layer
        self._cache = {}

    def fit(self, train, edges):          # trained offline in SpatialProp's env; nothing to do here
        return self

    def _pred(self, perturbation):
        if perturbation not in self._cache:
            self._cache[perturbation] = read_h5ad_layer(self.prediction_paths[perturbation],
                                                        self.layer)
        return self._cache[perturbation]

    def predict_niche(self, data, perturbation, edges):
        """SpatialProp's predicted bystander niche for `perturbation`: its predicted expression at the
        non-perturbed neighbours of every perturbed center. Same shape/role as the niche clouds in
        spbench.compare, so it slots straight into compare_to_baseline(extra=...)."""
        Xpred = self._pred(perturbation)
        centers = np.where(data.perturbation == perturbation)[0]
        nbs = []
        for c in centers:
            nb = neighbors_of(c, edges)
            nbs.append(nb[~data.is_perturbed[nb]])           # bystanders only
        if not nbs:
            return np.zeros((0, Xpred.shape[1]), float)
        return Xpred[np.concatenate(nbs)]

    def predict(self, perturbation, reference_cells, edges, center, neighbors):
        """EndToEndModel ABC: predicted expression at `neighbors` (from the cached SpatialProp output)."""
        return self._pred(perturbation)[np.asarray(neighbors)]
