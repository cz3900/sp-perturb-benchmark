import numpy as np
from .base import Metric
from . import register


class CompL1(Metric):
    """D3 composition metric: total-variation distance between the mean niche cell-type
    composition of pred and gt. Inputs are (n_niches, C) stacks of simplex rows (each row a
    composition over cell types). We score the GROUP-MEAN composition (population-average, like
    MSE / PCC-delta), then return TV = 0.5 * sum|p_pred - p_obs| in [0, 1]. Lower is better.

    Default choice for Phase 1: TV (= L1/2) is robust to exact zeros in the simplex (absent cell
    types) and needs no smoothing, unlike JS / cross-entropy / Aitchison. Phase 2 (G7/G9) revisits
    JS / cross-entropy / Aitchison once the observed-composition line is in place."""
    name = "comp_l1"
    higher_is_better = False
    status = "planned"

    def compute(self, pred, gt, context=None) -> float:
        pred = np.asarray(pred, float); gt = np.asarray(gt, float)
        if len(pred) == 0 or len(gt) == 0:
            return float("nan")
        return float(0.5 * np.sum(np.abs(pred.mean(0) - gt.mean(0))))


register(CompL1())
