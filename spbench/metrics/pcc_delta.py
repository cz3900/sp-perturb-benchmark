import numpy as np
from .base import Metric
from . import register

def delta_corr(d_pred, d_true) -> float:
    """Pearson correlation between two gene-wise shift vectors. NaN if either is essentially flat
    (no direction to correlate)."""
    d_pred = np.asarray(d_pred, float); d_true = np.asarray(d_true, float)
    if d_pred.std() < 1e-12 or d_true.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(d_pred, d_true)[0, 1])

class PccDelta(Metric):
    """PCC-delta (GEARS/scGPT convention): Pearson correlation between the predicted and the true
    gene-wise *shift* (delta = group mean - reference mean). Measures whether the prediction moves
    the right genes in the right direction. Bounded [-1, 1], self-anchored at 0 (a no-effect
    prediction has a flat delta -> NaN -> no directional skill).

    pred / gt are cell groups; the 'no effect' reference is passed via
    context={'reference': <cell array>} (defaults to the origin if absent)."""
    name = "pcc_delta"
    higher_is_better = True

    def compute(self, pred, gt, context=None) -> float:
        pred = np.asarray(pred, float); gt = np.asarray(gt, float)
        if len(pred) == 0 or len(gt) == 0:
            return float("nan")
        ref = None if context is None else context.get("reference")
        ref = np.asarray(ref, float) if ref is not None else None
        ref_mean = ref.mean(0) if ref is not None and len(ref) else np.zeros(pred.shape[1])
        return delta_corr(pred.mean(0) - ref_mean, gt.mean(0) - ref_mean)

register(PccDelta())
