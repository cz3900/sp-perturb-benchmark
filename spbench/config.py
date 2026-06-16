import yaml
import numpy as np
from .graph import build_knn_graph
from .harness import fill_2x2, _control_reference
from .judge import attribute, leakage_pass
from .models.trivial_seed import TrivialSeed
from .models.gaussian_prop import GaussianProp
from .models.gcn_prop import SimpleGCN
from .adapters import get_adapter


def run_benchmark(data, perturbations=None, k=15, k_ref=5, gcn_kwargs=None, progress=True):
    """Run the MVP benchmark on a StandardData. Returns grids + attribution + leakage flags."""
    gcn_kwargs = gcn_kwargs or {}
    edges = build_knn_graph(data, k=k)
    seed = TrivialSeed().fit(data)
    base = GaussianProp().fit(data, edges)
    learned = SimpleGCN(**gcn_kwargs).fit(data, edges)
    perturbations = perturbations or data.perturbations()
    X_ref = _control_reference(data)   # identical across perturbations -> compute once
    grids, attrib, leak = {}, {}, {}
    _bar = perturbations
    if progress:
        try:
            from tqdm.auto import tqdm
            _bar = tqdm(perturbations, desc="benchmark")
        except Exception:
            _bar = perturbations
    for p in _bar:
        g = fill_2x2(data, p, edges, seed, base, learned, k_ref=k_ref, X_ref=X_ref)
        grids[p] = g
        attrib[p] = attribute(g)
        leak[p] = leakage_pass(g)
    ranking = sorted(perturbations, key=lambda p: attrib[p]["end_to_end"])
    return {"grids": grids, "attribution": attrib, "leakage_pass": leak, "ranking": ranking}


def run_from_yaml(path: str):
    cfg = yaml.safe_load(open(path))
    data = get_adapter(cfg["adapter"])(**cfg.get("adapter_kwargs", {})).load()
    return run_benchmark(data, perturbations=cfg.get("perturbations"),
                         k=cfg.get("k", 15), k_ref=cfg.get("k_ref", 5),
                         gcn_kwargs=cfg.get("gcn_kwargs", {}))
