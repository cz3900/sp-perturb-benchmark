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

def _control_residuals(data):
    """Per-cell-type pool of CONTROL residuals (X_control - its cell-type control mean), plus a
    global pool fallback under key None.

    A propagation model emits one vector per cell — the conditional *mean*. The observed niche is
    a full-variance cloud (spread ~6), so scoring a near-degenerate mean-field cloud with the
    energy distance (a *distributional* metric) inflates it structurally, regardless of whether
    the predicted shift is right. Adding a sampled control residual to each predicted cell is the
    deterministic analogue of a generative model drawing per-cell samples around its mean: it
    restores realistic per-cell biological variance without moving the mean, so the energy
    distance measures the predicted SHIFT fairly. Residuals come only from CONTROL cells, never
    from the observed perturbed niche, so they cannot leak."""
    ctrl = data.control_pool
    pools = {}
    gmean = data.X[ctrl].mean(0) if ctrl.any() else data.X.mean(0)
    pools[None] = (data.X[ctrl] - gmean) if ctrl.any() else (data.X - gmean)
    for ct in np.unique(data.cell_type):
        m = ctrl & (data.cell_type == ct)
        if m.any():
            pools[ct] = data.X[m] - data.X[m].mean(0)
    return pools

def _draw_residuals(pools, cell_types, rng):
    """One sampled residual per cell, drawn from its cell-type pool (global pool as fallback)."""
    g = pools[None]
    out = np.empty((len(cell_types), g.shape[1]), float)
    for i, ct in enumerate(cell_types):
        R = pools.get(ct, g)
        if len(R) == 0:
            R = g
        out[i] = R[rng.integers(len(R))] if len(R) else 0.0
    return out

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


def _residuals_from(Xexpr, data):
    """Per-cell-type control residual pools in the space of `Xexpr` (+ global pool under None).
    Mirrors _control_residuals but in an explicit space (e.g. eval_X / scGEN log-norm)."""
    pool = data.control_pool
    pools = {None: (Xexpr[pool] - (Xexpr[pool].mean(0) if pool.any() else Xexpr.mean(0)))
             if pool.any() else (Xexpr - Xexpr.mean(0))}
    for ct in np.unique(data.cell_type):
        m = pool & (data.cell_type == ct)
        if m.any():
            pools[ct] = Xexpr[m] - Xexpr[m].mean(0)
    return pools


def fill_2x2(data, perturbation, edges, seed_model, baseline_prop, learned_prop, k_ref=5,
             X_ref=None, return_niches=False, residuals=None, noise_seed=0, eval_X=None):
    """Fill the seed×propagation 2×2 for one perturbation.
    Rows = {GT seed, Model seed}, Cols = {baseline prop, learned prop}.
    Each cell scores propagation E-distance vs the observed perturbed-niche distribution.

    `residuals` (from `_control_residuals`) gives every predicted cell realistic per-cell
    variance so the energy distance compares the predicted *shift* fairly instead of penalising
    the variance collapse of a mean-field prediction (see `_control_residuals`). The residual
    draws are reseeded identically for each of the four cells, so e1..e4 differ only by their
    mean field. Pass residuals=None to score the raw mean-only predictions.

    With return_niches=True the grid also carries `_niches` = {observed, reference, "1".."4"}
    (the predicted bystander-niche arrays for all four 2x2 cells) so each can be compared to the
    no-effect baseline downstream via spbench.compare."""
    energy = get_metric("energy")
    centers = np.where(data.perturbation == perturbation)[0]
    gt = propagation_gt(data, perturbation, edges, k_ref=k_ref)

    # niche scoring space: when eval_X is the (n_cells, G) co-scoring matrix (e.g. scGEN's log-norm
    # space) build the WHOLE niche path in it, so a model whose seed lives there (scGEN) is scored
    # consistently — observed/reference niche, the control X_ref, and the residual pools all come
    # from eval_X. Otherwise everything stays in data.X (the default baseline path, unchanged).
    cospace = isinstance(eval_X, np.ndarray)
    Xn = np.asarray(eval_X, float) if cospace else data.X
    observed = Xn[gt["pert_nb"]]

    refs = control_reference_centers(data, centers)   # aggregate control: same-cell-type control cells, no feature match
    if cospace:
        X_ref = _xref_from(Xn, data)
        # recompute the residual pool in Xn space (the passed-in one is data.X space), but only when
        # the caller actually wants a distributional readout (residuals is not None).
        res_pool = _residuals_from(Xn, data) if residuals is not None else None
    else:
        if X_ref is None:
            X_ref = _control_reference(data)   # propagation starts from the control niche, NOT the observed one
        res_pool = residuals

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
        rng = np.random.default_rng(noise_seed)   # identical residual draws across the 4 cells
        preds = []
        for c, rc in zip(centers, refs):
            nb = _bystanders(data, c, edges)
            if len(nb) == 0:
                continue
            if use_gt_seed:
                seed_state = Xn[c]                                         # oracle: true perturbed center (in scoring space)
            elif per_center_seed is not None:
                seed_state = per_center_seed[int(c)]                       # this center's OWN cached row (already in eval_X space)
            else:
                # model seed predicts from the ALL same-cell-type CONTROL cells (aggregate control,
                # no feature matching), never the center's own value -> one seed per cell type
                seed_state = seed_model.predict_seed(perturbation, data.X[rc]).mean(0)
            pred = prop_model.propagate(X_ref, edges, c, seed_state, nb)
            if res_pool is not None:                                       # distributional readout
                pred = pred + _draw_residuals(res_pool, data.cell_type[nb], rng)
            preds.append(pred)
        return np.vstack(preds) if preds else np.zeros((0, Xn.shape[1]))

    cells = {
        "1": collect(True, baseline_prop),
        "2": collect(True, learned_prop),
        "3": collect(False, baseline_prop),
        "4": collect(False, learned_prop),
    }
    grid = {k: {"energy_prop": energy.compute(v, observed)} for k, v in cells.items()}
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
        # variance-restored seed for the ENERGY readout (pcc/mse stay on the raw mean seed): mirror
        # the niche path's control residual so a collapsed mean-field seed isn't structurally
        # penalised by the energy distance. data.X space only — when eval_X is an ndarray (scGEN
        # log-norm) the seed already carries variance in its own space and the data.X residual pool
        # would be the wrong space.
        if residuals is not None and not isinstance(eval_X, np.ndarray) and len(seed_pred):
            _rng_s = np.random.default_rng(noise_seed + 7)
            seed_pred_resid = seed_pred + _draw_residuals(residuals, data.cell_type[centers], _rng_s)
        else:
            seed_pred_resid = seed_pred
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
        grid["_niches"] = {"observed": observed, "reference": Xn[gt["ref_nb"]],   # reference co-spaced with observed
                           "1": cells["1"], "2": cells["2"], "3": cells["3"], "4": cells["4"],
                           "seed_obs": seed_obs, "seed_pred": seed_pred,
                           "seed_pred_resid": seed_pred_resid,
                           "seed_ref": seed_ref,
                           "eval_X": carried_eval_X}   # transform carried downstream (callable/None)
    return grid
