from abc import ABC, abstractmethod
import numpy as np

class SeedModel(ABC):
    """Predicts the perturbed CENTER-cell distribution (seed) for a perturbation."""
    name = "seed"
    @abstractmethod
    def fit(self, train) -> "SeedModel": ...
    @abstractmethod
    def predict_seed(self, perturbation: str, reference_cells: np.ndarray) -> np.ndarray:
        """reference_cells: (m, n_genes) control starting states. Returns (m, n_genes)."""

class PropModel(ABC):
    """Given perturbed center states on a graph, predicts neighbour (propagation) states."""
    name = "prop"
    @abstractmethod
    def fit(self, train, edges) -> "PropModel": ...
    @abstractmethod
    def propagate(self, X_reference: np.ndarray, edges: np.ndarray,
                  center: int, seed_state: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
        """Returns predicted (len(neighbors), n_genes) for the bystander neighbours."""

class EndToEndModel(ABC):
    name = "endtoend"
    @abstractmethod
    def fit(self, train, edges) -> "EndToEndModel": ...
    @abstractmethod
    def predict(self, perturbation, reference_cells, edges, center, neighbors): ...
