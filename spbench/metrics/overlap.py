from .base import Metric
from . import register, get_metric


class Overlap(Metric):
    """Niche cell-type composition Overlap = 1 - TV (total-variation distance), the composition
    peer of pcc_delta. Inputs are (n, C) stacks of simplex rows; scores the group-mean composition
    (like pcc_delta scores the group-mean shift). 1 = identical composition, 0 = disjoint; read as
    'what fraction of the niche cell-type composition is predicted right'. Higher is better.

    Delegates the TV computation to comp_l1 so there is exactly one definition of the distance."""
    name = "overlap"
    higher_is_better = True
    status = "active"

    def compute(self, pred, gt, context=None) -> float:
        tv = get_metric("comp_l1").compute(pred, gt, context)
        return float("nan") if tv != tv else 1.0 - tv


register(Overlap())
