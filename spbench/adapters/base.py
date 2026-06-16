from abc import ABC, abstractmethod

class DatasetAdapter(ABC):
    """Normalizes any dataset/format into a single StandardData."""
    @abstractmethod
    def load(self): ...
