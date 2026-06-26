from .base import DatasetAdapter
from .saunders import SaundersAdapter
from .dhainaut import DhainautAdapter
from .cheng import ChengAdapter
from .shen import ShenAdapter
from .binan import BinanTumorsAdapter
from .spac_seq import SpacSeqAdapter

_REGISTRY = {"saunders": SaundersAdapter, "dhainaut": DhainautAdapter,
             "cheng": ChengAdapter, "shen": ShenAdapter, "binan_tumors": BinanTumorsAdapter,
             "spac_seq": SpacSeqAdapter}

def get_adapter(name: str):
    return _REGISTRY[name]

from .scgen_export import build_lognorm_X, export_to_scgen_h5
