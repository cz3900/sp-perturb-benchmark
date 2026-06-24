from .base import DatasetAdapter
from .saunders import SaundersAdapter

_REGISTRY = {"saunders": SaundersAdapter}

def get_adapter(name: str):
    return _REGISTRY[name]

from .scgen_export import build_lognorm_X, export_to_scgen_h5
