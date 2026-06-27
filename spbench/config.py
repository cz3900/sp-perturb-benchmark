import yaml
import numpy as np
from .graph import build_knn_graph
from .harness import fill_2x2, _control_reference, _control_reference_aggregate
from .judge import attribute, leakage_pass
from .compare import compare_to_baseline, evaluate_seed
from .permutation import permutation_null
from .models.trivial_seed import TrivialSeed
from .models.gaussian_prop import GaussianProp
from .models.gcn_prop import SimpleGCN
from .adapters import get_adapter


def run_benchmark(data, perturbations=None, k=15, k_ref=5, gcn_kwargs=None, progress=True,
                  compare=True, n_perm=None, perm_seed=0,
                  external_models=None, composition=False, annotator=None):
    """Run the MVP benchmark on a StandardData. Returns grids + attribution + leakage flags, and
    (compare=True) a niche PCC-delta of every 2x2 cell vs the no-effect baseline per perturbation:
    res['compare'][p] = {'pcc': {method: PCC-delta to observed}, 'mag': {method: relative shift
    size}, 'n'}. pcc > 0 means the method moves the niche in the right direction; pcc['null'] is
    NaN (a flat no-effect shift has no direction)."""
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
    grids, attrib, leak, cmp, seed_eval, comp_board = {}, {}, {}, {}, {}, {}
    _bar = perturbations
    if progress:
        try:
            from tqdm.auto import tqdm
            _bar = tqdm(perturbations, desc="benchmark")
        except Exception:
            _bar = perturbations
    for p in _bar:
        niches = None
        g = fill_2x2(data, p, edges, seed, base, learned, k_ref=k_ref, X_ref=X_ref,
                     return_niches=compare)
        if compare and "_niches" in g:
            niches = g.pop("_niches")
            eval_X = niches.get("eval_X")                               # unified scoring-space transform
            # External / end-to-end models (CONCERT-style): their predicted bystander niche is
            # scored on the SAME PCC-delta footing as the 2x2 cells via extra=.
            extra = ({nm: m.predict_niche(data, p, edges) for nm, m in external_models.items()}
                     if external_models else None)
            cmp[p] = compare_to_baseline(niches, eval_X=eval_X, extra=extra,
                                         annotator=annotator)  # niche: PCC-delta + mag + Overlap
            seed_eval[p] = evaluate_seed(niches, eval_X=eval_X)         # seed: PCC-delta + MSE (direct)
        if composition:
            # D3 Overlap board. Observed gt vs no-change baseline is always available (native
            # labels when annotator is None); with an annotator the deployable model's predicted
            # niche expression is labelled by the SAME frozen instrument and scored too.
            from .composition import composition_eval, DEPLOYABLE
            pred_niches = ({DEPLOYABLE: niches["4"]}
                           if (annotator is not None and niches is not None and "4" in niches)
                           else None)
            comp_board[p] = composition_eval(data, p, edges, annotator=annotator,
                                             pred_niches=pred_niches)
        grids[p] = g
        attrib[p] = attribute(g)
        leak[p] = leakage_pass(g)
    ranking = sorted(perturbations, key=lambda p: attrib[p]["end_to_end"], reverse=True)
    res = {"grids": grids, "attribution": attrib, "leakage_pass": leak, "ranking": ranking}
    if compare:
        res["compare"] = cmp
        res["seed"] = seed_eval
    if composition:
        res["composition"] = comp_board
    if n_perm is not None:
        # Empirical permutation null per perturbation (Plan 3): reuse the SAME kNN `edges` built
        # above so the niche graph is identical to the rest of the benchmark.
        res["perm"] = {p: permutation_null(data, p, edges, n_perm=n_perm, seed=perm_seed)
                       for p in perturbations}
    return res


def run_from_yaml(path: str):
    cfg = yaml.safe_load(open(path))
    data = get_adapter(cfg["adapter"])(**cfg.get("adapter_kwargs", {})).load()
    # Opt-in unified cell annotation (reserved modification point): relabel data.cell_type
    # with cfg['annotator'] before scoring. No 'annotator' key -> unchanged (back-compat).
    from .annotators import annotate_from_config
    data, annotator = annotate_from_config(data, cfg)
    return run_benchmark(data, perturbations=cfg.get("perturbations"),
                         k=cfg.get("k", 15), k_ref=cfg.get("k_ref", 5),
                         gcn_kwargs=cfg.get("gcn_kwargs", {}),
                         composition=cfg.get("composition", False), annotator=annotator)
