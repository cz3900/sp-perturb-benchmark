from .base import Metric

_REGISTRY: dict[str, Metric] = {}

def register(metric: Metric):
    _REGISTRY[metric.name] = metric
    return metric

def get_metric(name: str) -> Metric:
    return _REGISTRY[name]

def list_metrics() -> list:
    return sorted(_REGISTRY)

from . import energy, rho_niche, moran  # noqa: E402  (self-register)
