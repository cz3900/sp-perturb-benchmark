import numpy as np
from .graph import neighbors_of
from .propagation_gt import propagation_gt
from .metrics import get_metric
from .reference_aggregate import aggregate_control, control_reference_centers

def _bystanders(data, center, edges):
    nb = neighbors_of(center, edges)
    return nb[~data.is_perturbed[nb]]

def _control_reference(data):
    """Reference ('unperturbed') state matrix: each cell -> mean expression of CONTROL cells of
    its cell type (global control mean as fallback). Propagation starts from this instead of the
    observed matrix, so a prediction can never trivially reproduce the observed perturbed niche
    (which would be leakage). With a control reference the seed shift = perturbed - control is
    real, and the predicted niche is genuinely compared against the observed perturbed niche."""
    ctrl = data.control_pool
    global_mean = data.X[ctrl].mean(0) if ctrl.any() else data.X.mean(0)
    X_ref = np.tile(global_mean, (data.n_cells, 1)).astype(float)
    for ct in np.unique(data.cell_type):
        m = ctrl & (data.cell_type == ct)
        if m.any():
            X_ref[data.cell_type == ct] = data.X[m].mean(0)
    return X_ref

def _control_reference_aggregate(data, edges):
    """Aggregate-control version of `_control_reference` (G1). Builds the per-cell reference
    matrix by broadcasting each cell_type's CONTROL-cell aggregate mean (from
    `reference_aggregate.aggregate_control`) back to every cell of that type. Same numbers as
    `_control_reference`, but sourced through the aggregate-control path so the new pipeline
    never touches matched-control feature-space pairing. The legacy `_control_reference` is kept
    for fallback / regression comparison."""
    agg = aggregate_control(data, edges)
    X_ref = np.empty((data.n_cells, data.n_genes), float)
    for ct in agg.cell_types:
        X_ref[data.cell_type == ct] = agg.expr[ct]
    return X_ref

def _xref_from(Xexpr, data):
    """Per-cell control-reference matrix in the space of `Xexpr`: each cell -> mean of Xexpr over
    control-pool cells of its cell type (global control mean fallback). Mirrors
    _control_reference_aggregate but in an explicit space (e.g. eval_X / scGEN log-norm)."""
    pool = data.control_pool
    gmean = Xexpr[pool].mean(0) if pool.any() else Xexpr.mean(0)
    Xr = np.tile(gmean, (Xexpr.shape[0], 1)).astype(float)
    for ct in np.unique(data.cell_type):
        m = pool & (data.cell_type == ct)
        if m.any():
            Xr[data.cell_type == ct] = Xexpr[m].mean(0)
    return Xr


def _pcc_prop(pred, observed, reference):
    """Niche PCC-delta of a 2x2 cell's predicted bystander cloud vs the observed perturbed niche,
    with the reference (no-effect) niche as the shift baseline. Higher = better (right direction).
    The 2x2 grid score: it replaces the retired energy distance — same per-cell structure, but a
    mean-based, bounded, direction metric."""
    return get_metric("pcc_delta").compute(pred, observed, {"reference": reference})


def fill_2x2(data, perturbation, edges, seed_model, baseline_prop, learned_prop, k_ref=5,
             X_ref=None, return_niches=False, noise_seed=0, eval_X=None):
    """Fill the seed×propagation 2×2 for one perturbation.
    Rows = {GT seed, Model seed}, Cols = {baseline prop, learned prop}.
    Each cell scores its propagated bystander cloud vs the observed perturbed-niche distribution
    with a niche PCC-delta (`pcc_prop`, higher = better, reference niche as the shift baseline).

    With return_niches=True the grid also carries `_niches` = {observed, reference, "1".."4"}
    (the predicted bystander-niche arrays for all four 2x2 cells) so each can be compared to the
    no-effect baseline downstream via spbench.compare."""
    centers = np.where(data.perturbation == perturbation)[0]
    gt = propagation_gt(data, perturbation, edges, k_ref=k_ref)

    # niche scoring space: when eval_X is the (n_cells, G) co-scoring matrix (e.g. scGEN's log-norm
    # space) build the WHOLE niche path in it, so a model whose seed lives there (scGEN) is scored
    # consistently — observed/reference niche and the control X_ref all come from eval_X. Otherwise
    # everything stays in data.X (the default baseline path, unchanged).
    cospace = isinstance(eval_X, np.ndarray)
    Xn = np.asarray(eval_X, float) if cospace else data.X
    observed = Xn[gt["pert_nb"]]
    reference = Xn[gt["ref_nb"]]

    refs = control_reference_centers(data, centers)   # aggregate control: same-cell-type control cells, no feature match
    if cospace:
        X_ref = _xref_from(Xn, data)
    else:
        if X_ref is None:
            X_ref = _control_reference(data)   # propagation starts from the control niche, NOT the observed one

    # ScgenSeedModel (and any loader exposing `.centers()`) caches a per-center-ALIGNED
    # (n_centers, G) seed array; predict_seed IGNORES reference_cells and returns the WHOLE array,
    # so the per-`rc` `predict_seed(...).mean(0)` path below would collapse it to one global-mean
    # vector broadcast to every center, destroying the per-center alignment (the whole point of
    # /obs/center_idx). When the model is per-center-aligned, build a {center_idx -> its own cached
    # row} map and feed EACH center its OWN row as the seed. Ordinary seed models (TrivialSeed etc.,
    # no `.centers()`) keep the unchanged matched-control `predict_seed(...).mean(0)` behavior.
    per_center_seed = None
    if hasattr(seed_model, "centers"):
        cached = np.asarray(seed_model.predict_seed(perturbation, data.X), float)  # (n_centers, G)
        cidx = np.asarray(seed_model.centers(perturbation))                        # aligns cached rows
        row_of = {int(ci): i for i, ci in enumerate(cidx)}
        per_center_seed = {int(c): cached[row_of[int(c)]] for c in centers}

    def collect(use_gt_seed, prop_model):
        preds = []
        for c, rc in zip(centers, refs):
            nb = _bystanders(data, c, edges)
            if len(nb) == 0:
                continue
            if use_gt_seed:
                seed_state = Xn[c]                                         # GT seed: true perturbed center (in scoring space)
            elif per_center_seed is not None:
                seed_state = per_center_seed[int(c)]                       # this center's OWN cached row (already in eval_X space)
            else:
                # model seed predicts from the ALL same-cell-type CONTROL cells (aggregate control,
                # no feature matching), never the center's own value -> one seed per cell type
                seed_state = seed_model.predict_seed(perturbation, data.X[rc]).mean(0)
            pred = prop_model.propagate(X_ref, edges, c, seed_state, nb)
            preds.append(pred)
        return np.vstack(preds) if preds else np.zeros((0, Xn.shape[1]))

    cells = {
        "1": collect(True, baseline_prop),
        "2": collect(True, learned_prop),
        "3": collect(False, baseline_prop),
        "4": collect(False, learned_prop),
    }
    grid = {k: {"pcc_prop": _pcc_prop(v, observed, reference)} for k, v in cells.items()}
    if return_niches:
        # seed evaluation data: model-seed prediction vs the observed perturbed centers, with the
        # matched control cells as the shift baseline (scored directly, not through the niche).
        # For per-center-aligned loaders, seed_pred is each center's OWN cached row (same alignment
        # as the propagation loop); otherwise the per-`rc` matched-control mean (unchanged).
        if not len(centers):
            seed_pred = np.zeros((0, data.n_genes))
        elif per_center_seed is not None:
            seed_pred = np.array([per_center_seed[int(c)] for c in centers])
        else:
            seed_pred = np.array([seed_model.predict_seed(perturbation, data.X[rc]).mean(0)
                                  for rc in refs])
        seed_ref_idx = np.unique(np.concatenate(refs)) if len(refs) else np.array([], int)
        # eval_X is dual-semantic (cross-task convention #1, pcc_delta is NOT space-robust):
        #   - np.ndarray (G6 scGEN log-norm matrix): the model's seed_pred already lives in this
        #     space, so slice seed_obs/seed_ref into it HERE (eval_X[centers] / eval_X[seed_ref_idx])
        #     and carry eval_X=None downstream — the three are already co-spaced, no transform left.
        #   - callable (G4, e.g. np.arcsinh) or None: keep seed_obs/seed_ref in data.X space and
        #     stash the callable/None unchanged so evaluate_seed/compare_to_baseline apply it at
        #     scoring time (the G4 path). The two branches MUST NOT collide.
        if isinstance(eval_X, np.ndarray):
            eval_space = np.asarray(eval_X, float)
            seed_obs = eval_space[centers]
            seed_ref = eval_space[seed_ref_idx]
            carried_eval_X = None
        else:
            seed_obs = data.X[centers]
            seed_ref = data.X[seed_ref_idx]
            carried_eval_X = eval_X
        grid["_niches"] = {"observed": observed, "reference": reference,   # reference co-spaced with observed
                           "1": cells["1"], "2": cells["2"], "3": cells["3"], "4": cells["4"],
                           "seed_obs": seed_obs, "seed_pred": seed_pred,
                           "seed_ref": seed_ref,
                           "eval_X": carried_eval_X}   # transform carried downstream (callable/None)
    return grid
