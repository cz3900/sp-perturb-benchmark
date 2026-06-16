from .base import DatasetAdapter
from .saunders import SaundersAdapter

_REGISTRY = {"saunders": SaundersAdapter}

def get_adapter(name: str):
    return _REGISTRY[name]
