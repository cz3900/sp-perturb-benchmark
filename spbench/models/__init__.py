from .base import SeedModel, PropModel, EndToEndModel

_REGISTRY = {}

def register(cls):
    _REGISTRY[cls.name] = cls
    return cls

def get_model(name: str):
    return _REGISTRY[name]

def list_models() -> list:
    return sorted(_REGISTRY)

# auto-register built-in models when present (resilient during incremental build)
for _m in ("trivial_seed", "gaussian_prop", "gcn_prop"):
    try:
        __import__(f"{__name__}.{_m}")
    except ImportError:
        pass
