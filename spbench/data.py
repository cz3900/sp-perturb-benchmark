from dataclasses import dataclass, field
import numpy as np

CONTROL = "control"
UNLABELED = "none"

@dataclass
class StandardData:
    """Universal internal representation. Every adapter normalizes into this."""
    X: np.ndarray            # (n_cells, n_genes) expression
    coords: np.ndarray       # (n_cells, 2) spatial coords
    perturbation: np.ndarray # (n_cells,) str: gene symbol | 'control' | 'none'
    cell_type: np.ndarray    # (n_cells,) str
    batch: np.ndarray        # (n_cells,) str  slice/section id (graph is built within batch)
    gene_names: list
    meta: dict = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    @property
    def is_control(self) -> np.ndarray:
        return self.perturbation == CONTROL

    @property
    def is_unlabeled(self) -> np.ndarray:
        return self.perturbation == UNLABELED

    @property
    def is_perturbed(self) -> np.ndarray:
        return ~self.is_control & ~self.is_unlabeled

    def perturbations(self) -> list:
        return sorted(set(self.perturbation[self.is_perturbed]))

    def subset(self, idx: np.ndarray) -> "StandardData":
        return StandardData(
            X=self.X[idx], coords=self.coords[idx],
            perturbation=self.perturbation[idx], cell_type=self.cell_type[idx],
            batch=self.batch[idx], gene_names=list(self.gene_names), meta=dict(self.meta),
        )
