from abc import ABC, abstractmethod
import numpy as np

class Metric(ABC):
    name: str = "metric"
    higher_is_better: bool = False

    @abstractmethod
    def compute(self, pred: np.ndarray, gt: np.ndarray, context: dict | None = None) -> float:
        ...
