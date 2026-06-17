import numpy as np
from .base import Metric
from . import register

class MSE(Metric):
    """Mean squared error between the two groups' MEAN expression profiles (a population-average
    metric; no cell-to-cell pairing). Lower is better. Used for the magnitude of the seed/niche
    shift error, complementing the direction-only PCC-delta."""
    name = "mse"
    higher_is_better = False

    def compute(self, pred, gt, context=None) -> float:
        pred = np.asarray(pred, float); gt = np.asarray(gt, float)
        if len(pred) == 0 or len(gt) == 0:
            return float("nan")
        return float(np.mean((pred.mean(0) - gt.mean(0)) ** 2))

register(MSE())
