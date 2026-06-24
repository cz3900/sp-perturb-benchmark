import yaml
import numpy as np
from .graph import build_knn_graph
from .harness import fill_2x2, _control_reference, _control_reference_aggregate, _control_residuals
from .judge import attribute, leakage_pass
from .compare import compare_to_baseline, evaluate_seed
from .permutation import permutation_null
from .models.trivial_seed import TrivialSeed
from .models.gaussian_prop import GaussianProp
from .models.gcn_prop import SimpleGCN
from .adapters import get_adapter


def run_benchmark(data, perturbations=None, k=15, k_ref=5, gcn_kwargs=None, progress=True,
                  compare=True, distributional=True, n_perm=None, perm_seed=0,
                  external_models=None):
    """Run the MVP benchmark on a StandardData. Returns grids + attribution + leakage flags, and
    (compare=True) a comparison of every 2x2 cell to the no-effect baseline per perturbation:
    res['compare'][p] = {'e': {method: energy distance to observed}, 'gain': {method: e_null - e},
    'n', 'has_effect'}. gain > 0 means the method beats predicting 'the neighbours did not change'.

    distributional=True (default) gives each predicted cell realistic per-cell variance (sampled
    control residuals) before scoring, so the energy distance compares the predicted *shift* fairly
    rather than penalising the variance collapse of a mean-field prediction. Set False to score the
    raw mean-only predictions (variance-collapsed, energy distance structurally inflated)."""
    gcn_kwargs = gcn_kwargs or {}
    edges = build_knn_graph(data, k=k)
    seed = TrivialSeed().fit(data)
    base = GaussianProp().fit(data, edges)
    learned = SimpleGCN(**gcn_kwargs).fit(data, edges)
    perturbations = perturbations or data.perturbations()
    # Active control reference = sample-level aggregate-control path (G1), proven numerically
    # identical to the legacy `_control_reference` (tests/test_harness_aggregate_ref.py). It needs
    # the kNN `edges` graph (already built above) for its bystander-niche aggregation. The legacy
    # `_control_reference(data)` is kept in harness.py as the documented fallback.
    X_ref = _control_reference_aggregate(data, edges)   # identical across perturbations -> compute once
    residuals = _control_residuals(data) if distributional else None
    grids, attrib, leak, cmp, seed_eval = {}, {}, {}, {}, {}
    _bar = perturbations
    if progress:
        try:
            from tqdm.auto import tqdm
            _bar = tqdm(perturbations, desc="benchmark")
        except Exception:
            _bar = perturbations
    for p in _bar:
        g = fill_2x2(data, p, edges, seed, base, learned, k_ref=k_ref, X_ref=X_ref,
                     return_niches=compare, residuals=residuals)
        if compare and "_niches" in g:
            niches = g.pop("_niches")
            eval_X = niches.get("eval_X")                               # unified scoring-space transform
            # External / end-to-end models (CONCERT-style): their predicted bystander niche is
            # scored on the SAME matched-n / gain / PCC-delta footing as the 2x2 cells via extra=.
            extra = ({nm: m.predict_niche(data, p, edges) for nm, m in external_models.items()}
                     if external_models else None)
            cmp[p] = compare_to_baseline(niches, residuals=residuals, eval_X=eval_X,
                                         extra=extra)                   # niche: E-dist/gain + PCC-delta
            seed_eval[p] = evaluate_seed(niches, eval_X=eval_X)         # seed: PCC-delta + MSE (direct)
        grids[p] = g
        attrib[p] = attribute(g)
        leak[p] = leakage_pass(g)
    ranking = sorted(perturbations, key=lambda p: attrib[p]["end_to_end"])
    res = {"grids": grids, "attribution": attrib, "leakage_pass": leak, "ranking": ranking}
    if compare:
        res["compare"] = cmp
        res["seed"] = seed_eval
    if n_perm is not None:
        # Empirical permutation null per perturbation (Plan 3): reuse the SAME kNN `edges` built
        # above so the niche graph is identical to the rest of the benchmark.
        res["perm"] = {p: permutation_null(data, p, edges, n_perm=n_perm, seed=perm_seed)
                       for p in perturbations}
    return res


def run_from_yaml(path: str):
    cfg = yaml.safe_load(open(path))
    data = get_adapter(cfg["adapter"])(**cfg.get("adapter_kwargs", {})).load()
    return run_benchmark(data, perturbations=cfg.get("perturbations"),
                         k=cfg.get("k", 15), k_ref=cfg.get("k_ref", 5),
                         gcn_kwargs=cfg.get("gcn_kwargs", {}))
