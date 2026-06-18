"""CONCERT as a black-box, offline end-to-end model in our benchmark.

CONCERT trains + predicts on a GPU in its own conda env (incompatible with spbench's env), so we do
NOT import it here. Instead CONCERT is run offline (export StandardData -> CONCERT .h5, train,
`--stage eval` to flip cells to a perturbation), producing a counterfactual `.h5ad` of predicted
expression for every cell. This wrapper is a LOADER: it reads that `.h5ad` and extracts the
predicted *bystander niche* (neighbours of the perturbed centers), which scores in spbench.compare
exactly like our baselines' predicted niches. One `.h5ad` per evaluated perturbation.
"""
import numpy as np
import h5py
from .base import EndToEndModel
from ..graph import neighbors_of


def read_h5ad_X(path):
    """Read the (N, G) expression matrix from an `.h5ad` via h5py — no anndata dependency.
    Handles a dense `/X` dataset and a CSR/CSC-encoded `/X` group."""
    with h5py.File(path, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Dataset):
            return np.asarray(x, float)
        from scipy.sparse import csr_matrix, csc_matrix
        data = np.asarray(x["data"]); indices = np.asarray(x["indices"]); indptr = np.asarray(x["indptr"])
        shape = tuple(int(s) for s in x.attrs["shape"])
        enc = x.attrs.get("encoding-type", "csr_matrix")
        M = (csc_matrix if "csc" in str(enc) else csr_matrix)((data, indices, indptr), shape=shape)
        return M.toarray().astype(float)


class ConcertModel(EndToEndModel):
    """Loader wrapper around CONCERT's offline counterfactual predictions.

    prediction_paths : {perturbation: path to its CONCERT counterfactual `.h5ad`}.
    """
    name = "concert"

    def __init__(self, prediction_paths):
        self.prediction_paths = dict(prediction_paths)
        self._cache = {}

    def fit(self, train, edges):          # trained offline on the GPU; nothing to do here
        return self

    def _pred(self, perturbation):
        if perturbation not in self._cache:
            self._cache[perturbation] = read_h5ad_X(self.prediction_paths[perturbation])
        return self._cache[perturbation]

    def predict_niche(self, data, perturbation, edges):
        """CONCERT's predicted bystander niche for `perturbation`: its predicted expression at the
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
        """EndToEndModel ABC: predicted expression at `neighbors` (from the cached CONCERT output)."""
        return self._pred(perturbation)[np.asarray(neighbors)]
