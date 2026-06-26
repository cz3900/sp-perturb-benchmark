import numpy as np
from .base import Annotator
from . import register


@register
class MarkerAnnotator(Annotator):
    """Reference-free marker-score annotator (Phase-1 default, chosen over supervised
    CellTypist/scANVI). Pure numpy, no scanpy dependency.

    Why marker for THIS benchmark: one marker rule is identical across datasets and across
    gt/pred (true unification); it degrades gracefully on model-predicted (off-manifold)
    expression where a learned classifier would be unreliable; it is deterministic with no
    training randomness or leakage.

    Marker dictionary, two sources (auto-routed in `fit`):
      - expert labels present -> derive a data-driven dict (top genes per type by standardized
        mean difference, a lightweight rank_genes_groups), so the labels are not wasted;
      - no labels -> use an external dict from context['marker_dict'] (literature / PanglaoDB).

    Frozen instrument: per-gene mean/std from the reference are stored at `fit` and applied to
    EVERY matrix at `predict`, so the expression->label mapping is one fixed function shared by
    observed and predicted cells.
    """
    name = "marker"

    def __init__(self, n_markers: int = 20):
        self.n_markers = int(n_markers)

    def fit(self, ref_X, ref_labels=None, gene_names=None, context=None) -> "MarkerAnnotator":
        X = np.asarray(ref_X, dtype=float)
        self.gene_names_ = list(gene_names) if gene_names is not None else list(range(X.shape[1]))
        # frozen standardization stats (avoid div-by-zero on constant genes)
        self.mu_ = X.mean(0)
        self.sigma_ = X.std(0)
        self.sigma_[self.sigma_ == 0] = 1.0
        if ref_labels is not None:
            self.markers_ = self._derive_markers(X, np.asarray(ref_labels, dtype=object))
        elif context and "marker_dict" in context:
            self.markers_ = self._dict_to_idx(context["marker_dict"])
        else:
            raise ValueError("marker annotator needs ref_labels or context['marker_dict']")
        self.cats_ = np.array(sorted(self.markers_), dtype=object)
        return self

    def _zscore(self, X) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mu_) / self.sigma_

    def _derive_markers(self, X, labels) -> dict:
        Z = self._zscore(X)
        k = min(self.n_markers, X.shape[1])
        markers = {}
        for t in sorted(set(labels.tolist())):
            m = labels == t
            score = Z[m].mean(0) - (Z[~m].mean(0) if (~m).any() else 0.0)
            markers[t] = np.argsort(score)[::-1][:k].astype(int)
        return markers

    def _dict_to_idx(self, marker_dict) -> dict:
        name2i = {g: i for i, g in enumerate(self.gene_names_)}
        out = {}
        for t, genes in marker_dict.items():
            idx = [name2i[g] for g in genes if g in name2i]
            if idx:
                out[t] = np.asarray(idx, dtype=int)
        if not out:
            raise ValueError("no marker genes matched gene_names")
        return out

    def predict(self, X, gene_names=None, context=None) -> np.ndarray:
        Z = self._zscore(X)
        scores = np.column_stack([Z[:, self.markers_[t]].mean(1) for t in self.cats_])
        return self.cats_[np.argmax(scores, axis=1)]
