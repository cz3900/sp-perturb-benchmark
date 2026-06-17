import yaml
import numpy as np
from .graph import build_knn_graph
from .harness import fill_2x2, _control_reference
from .judge import attribute, leakage_pass
from .calibrate import calibrate_edistance, edist_matched, skill_score
from .models.trivial_seed import TrivialSeed
from .models.gaussian_prop import GaussianProp
from .models.gcn_prop import SimpleGCN
from .adapters import get_adapter


def _skill_for(niches):
    """Calibrate one perturbation and return baseline/learned skill (NaN if no signal)."""
    obs, ref = niches["observed"], niches["reference"]
    if len(obs) < 6 or len(ref) < 3:                       # too few cells to calibrate
        return {"has_signal": False, "baseline": float("nan"), "learned": float("nan"),
                "calibration": None}
    cal = calibrate_edistance(obs, ref)
    out = {"has_signal": cal["has_signal"], "calibration": cal}
    if cal["has_signal"]:
        for col, name in [("3", "baseline"), ("4", "learned")]:
            err, _ = edist_matched(niches[col], obs, cal["n"])
            out[name] = skill_score(err, cal)["skill"]
    else:
        out["baseline"] = out["learned"] = float("nan")
    return out


def run_benchmark(data, perturbations=None, k=15, k_ref=5, gcn_kwargs=None, progress=True,
                  compute_skill=True):
    """Run the MVP benchmark on a StandardData. Returns grids + attribution + leakage flags,
    and (compute_skill=True) a calibrated 0..1 skill per perturbation for the two deployable
    models (model-seed + baseline prop, model-seed + learned prop)."""
    gcn_kwargs = gcn_kwargs or {}
    edges = build_knn_graph(data, k=k)
    seed = TrivialSeed().fit(data)
    base = GaussianProp().fit(data, edges)
    learned = SimpleGCN(**gcn_kwargs).fit(data, edges)
    perturbations = perturbations or data.perturbations()
    X_ref = _control_reference(data)   # identical across perturbations -> compute once
    grids, attrib, leak, skill = {}, {}, {}, {}
    _bar = perturbations
    if progress:
        try:
            from tqdm.auto import tqdm
            _bar = tqdm(perturbations, desc="benchmark")
        except Exception:
            _bar = perturbations
    for p in _bar:
        g = fill_2x2(data, p, edges, seed, base, learned, k_ref=k_ref, X_ref=X_ref,
                     return_niches=compute_skill)
        if compute_skill and "_niches" in g:
            skill[p] = _skill_for(g.pop("_niches"))        # pop arrays -> keep grid lightweight
        grids[p] = g
        attrib[p] = attribute(g)
        leak[p] = leakage_pass(g)
    ranking = sorted(perturbations, key=lambda p: attrib[p]["end_to_end"])
    res = {"grids": grids, "attribution": attrib, "leakage_pass": leak, "ranking": ranking}
    if compute_skill:
        res["skill"] = skill
    return res


def run_from_yaml(path: str):
    cfg = yaml.safe_load(open(path))
    data = get_adapter(cfg["adapter"])(**cfg.get("adapter_kwargs", {})).load()
    return run_benchmark(data, perturbations=cfg.get("perturbations"),
                         k=cfg.get("k", 15), k_ref=cfg.get("k_ref", 5),
                         gcn_kwargs=cfg.get("gcn_kwargs", {}))
