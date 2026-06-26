import numpy as np
from .base import Annotator
from . import register


@register
class GivenAnnotator(Annotator):
    """Passthrough baseline: echoes the labels it was fit on. Identity on the same cells,
    so ground-truth composition uses the dataset's native expert labels unchanged.

    It deliberately CANNOT label new cells (it only memorises a label vector), so it raises
    on any matrix whose row count differs from the reference. That is the honest signal that
    scoring expression-only model predictions needs a real annotator (e.g. `marker`)."""
    name = "given"

    def fit(self, ref_X, ref_labels=None, gene_names=None, context=None) -> "GivenAnnotator":
        if ref_labels is None:
            raise ValueError("given annotator needs ref_labels to passthrough")
        self.labels_ = np.asarray(ref_labels, dtype=object)
        return self

    def predict(self, X, gene_names=None, context=None) -> np.ndarray:
        if len(X) != len(self.labels_):
            raise ValueError(
                "given annotator cannot infer labels for new cells (row-count mismatch); "
                "use a real annotator such as 'marker' to score predicted expression")
        return self.labels_.copy()
