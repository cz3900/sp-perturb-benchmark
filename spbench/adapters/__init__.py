from .base import DatasetAdapter
from .saunders import SaundersAdapter
from .dhainaut import DhainautAdapter

_REGISTRY = {"saunders": SaundersAdapter, "dhainaut": DhainautAdapter}

def get_adapter(name: str):
    return _REGISTRY[name]

from .scgen_export import build_lognorm_X, export_to_scgen_h5
