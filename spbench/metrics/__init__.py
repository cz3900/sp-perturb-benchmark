from .base import Metric

_REGISTRY: dict[str, Metric] = {}

def register(metric: Metric):
    _REGISTRY[metric.name] = metric
    return metric

def get_metric(name: str) -> Metric:
    return _REGISTRY[name]

def list_metrics(active_only: bool = False) -> list:
    return sorted(n for n, m in _REGISTRY.items()
                  if not active_only or getattr(m, "status", "active") == "active")

from . import rho_niche, moran, pcc_delta, mse, comp_l1, overlap  # noqa: E402  (self-register)
