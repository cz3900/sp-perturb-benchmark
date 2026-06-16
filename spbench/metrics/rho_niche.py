import numpy as np
from .base import Metric
from . import register

class RhoNiche(Metric):
    """Cross-niche response correlation. Inputs are dicts {niche_label: mean_delta_vector}.
    Flattens (niche x gene) response profiles and returns Pearson r of pred vs true."""
    name = "rho_niche"
    higher_is_better = True

    def compute(self, pred: dict, gt: dict, context=None) -> float:
        keys = sorted(set(pred) & set(gt))
        if len(keys) == 0:
            return float("nan")
        p = np.concatenate([np.asarray(pred[k], float).ravel() for k in keys])
        t = np.concatenate([np.asarray(gt[k], float).ravel() for k in keys])
        if p.std() == 0 or t.std() == 0:
            return 0.0
        return float(np.corrcoef(p, t)[0, 1])

register(RhoNiche())
