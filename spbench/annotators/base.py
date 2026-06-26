from abc import ABC, abstractmethod
import numpy as np


class Annotator(ABC):
    """Maps expression -> cell-type labels. The single source of `data.cell_type`,
    so the niche-composition line (niche.compute_niche_composition -> comp_l1 / Overlap)
    is fully determined by which annotator is plugged in.

    Contract: `fit` ONCE on a reference (real cells), then the SAME frozen instance is
    `predict`-ed on BOTH observed expression (-> ground-truth composition) AND model-
    predicted expression (-> predicted composition). Reusing one frozen instance is what
    makes the comparison fair: the annotator's systematic bias cancels on both sides.
    """
    name: str = "annotator"
    status: str = "active"

    @abstractmethod
    def fit(self, ref_X, ref_labels=None, gene_names=None, context=None) -> "Annotator":
        ...

    @abstractmethod
    def predict(self, X, gene_names=None, context=None) -> np.ndarray:
        ...
