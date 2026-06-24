# spbench/adapters/internal_output.py
from typing import Optional
import numpy as np
from ..prediction import StandardPrediction


def _aggregate(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Collapse a model's per-row output to a single aggregated gene vector (G,).

    Accepts (rows, G) -> mean over rows, or an already-aggregated (G,) -> passthrough.
    Returns None for None input or an empty (0, G) array (no rows to aggregate)."""
    if arr is None:
        return None
    a = np.asarray(arr, float)
    if a.ndim == 1:
        return a
    if a.shape[0] == 0:
        return None
    return a.mean(0)


def _to_simplex(comp: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalize a composition vector (C,) to sum-1 (simplex). Passthrough if already
    sums to ~1. Raises ValueError on a zero/negative-mass vector (undefined composition)."""
    if comp is None:
        return None
    c = np.asarray(comp, float)
    s = c.sum()
    if s <= 0:
        raise ValueError("composition must have positive total mass; got sum=%r" % s)
    return c / s


def to_prediction(seed_out: Optional[np.ndarray] = None,
                  prop_out: Optional[np.ndarray] = None,
                  composition: Optional[np.ndarray] = None) -> StandardPrediction:
    """Internal output adapter: map seed/prop model native outputs to the {D1,D2,D3} contract.

    seed_out    : SeedModel.predict_seed output (m, G) or aggregated (G,) -> D1.
    prop_out    : PropModel.propagate output (len(neighbours), G) -> D2 (mean over neighbours).
    composition : niche cell-type composition (C,) -> D3 (normalized to simplex).
    """
    return StandardPrediction(
        d1=_aggregate(seed_out),
        d2=_aggregate(prop_out),
        d3=_to_simplex(composition),
    )
