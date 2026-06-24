# spbench/prediction.py
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Dimension contract (spec §2.2):
#   D1 = predicted cell's own expression (seed),  shape (G,)
#   D2 = niche bystander-neighbour aggregated expression, shape (G,)
#   D3 = niche cell-type composition (simplex),   shape (C,)
# Each dim is None when a model does not cover it (capability matrix, spec §2.5).


@dataclass
class StandardPrediction:
    """Universal three-dim output contract. Every output adapter normalizes a model's
    native output into this. Only the dims a model covers are filled; the rest stay None."""
    d1: Optional[np.ndarray] = None
    d2: Optional[np.ndarray] = None
    d3: Optional[np.ndarray] = None

    def covered_dims(self) -> tuple:
        """Names of the dims this prediction actually fills, in canonical D1<D2<D3 order."""
        out = []
        if self.d1 is not None:
            out.append("D1")
        if self.d2 is not None:
            out.append("D2")
        if self.d3 is not None:
            out.append("D3")
        return tuple(out)

    def as_dict(self) -> dict:
        """{name: array} for covered dims only (drops the None dims)."""
        m = {"D1": self.d1, "D2": self.d2, "D3": self.d3}
        return {k: v for k, v in m.items() if v is not None}
