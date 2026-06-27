"""D3 Overlap board: predict the post-perturbation niche cell-type composition.

The headline number is **Overlap = 1 - TV** (TV = total-variation distance of two cell-type
composition vectors), read as "what fraction of the niche composition is predicted right"; the
mismatch 1-Overlap is "what fraction of cells would have to be relabelled". TV reuses the existing
`comp_l1` metric so there is one definition of the distance.

Two layers, separated so each is independently testable:
  - `composition_board` is PURE label-space scoring: observed niche labels vs a 'no-change'
    baseline (the reference niche) vs any predicted-niche labels -> Overlap + gain over baseline.
  - `composition_eval` is the BRIDGE: it pulls the perturbed / reference bystander niches
    (propagation_gt) and turns expression into labels with the FROZEN annotator (so observed and
    predicted go through one instrument), or uses native `data.cell_type` when no annotator.

Baseline is 'predict no change' (the reference / control niche); a model is only useful if its
gain = Overlap_model - Overlap_null is > 0.
"""
import numpy as np

from .data import CONTROL
from .metrics import get_metric
from .propagation_gt import propagation_gt

DEPLOYABLE = "model+learned"   # method label for the deployed model's predicted niche (matches compare.DEPLOYABLE)


# --------------------------------------------------------------------------- #
# Descriptive niche-effect layer (per-guide observed effect, the ground truth) #
# --------------------------------------------------------------------------- #

def build_adjacency(edges, n_cells: int) -> dict:
    """(2, E) [src, dst] graph -> {node: neighbour-index array}. Built once so neighbour lookup is
    O(degree) instead of O(E) (the per-call cost of graph.neighbors_of), which matters for the
    per-guide permutation null (n_perm re-aggregations)."""
    src, dst = np.asarray(edges[0]), np.asarray(edges[1])
    order = np.argsort(src, kind="stable")
    src_s, dst_s = src[order], dst[order]
    adj: dict = {}
    if len(src_s):
        uniq, start = np.unique(src_s, return_index=True)
        bounds = list(start) + [len(src_s)]
        for i, u in enumerate(uniq):
            adj[int(u)] = dst_s[bounds[i]:bounds[i + 1]]
    return adj


def neighbor_composition(data, centers, adj, cats) -> np.ndarray:
    """Cell-type composition of ALL spatial neighbours of `centers` (decision: all neighbours, not
    just non-perturbed bystanders) -> (C,) simplex. centers: int cell indices; adj: build_adjacency."""
    centers = np.asarray(centers, dtype=int)
    if len(centers) == 0:
        return np.zeros(len(cats), dtype=float)
    parts = [adj[c] for c in centers if c in adj]
    nb = np.concatenate(parts) if parts else np.empty(0, dtype=int)
    return niche_composition(data.cell_type[nb], cats)


def composition_effect(data, perturbation, adj, cats=None, n_perm: int = 0, seed: int = 0) -> dict:
    """Observed niche cell-type effect of one guide, with TWO baselines (decision #1):
      - `overlap_ntc`    : Overlap(NTC-cell niche, KO-cell niche)  -> raw effect vs unperturbed;
      - `overlap_pooled` : Overlap(all-perturbed niche, KO niche)  -> guide-SPECIFIC (de-confounds
        the shared 'cells carrying a detected guide sit in different regions' axis seen in real data).
    Overlap = 1 - TV, so LOWER overlap = BIGGER niche shift. Niche = all spatial neighbours.

    The NTC reference is `data.control_pool` (is_control cells, else the 'none' fallback) so the
    baseline works across datasets without a literal 'control' guide label (Shen, Dhainaut, ...).

    n_perm > 0 adds a permutation p-value: shuffle this-guide vs NTC labels (keeping n_ko fixed),
    recompute Overlap_ntc, p = fraction of permutations whose shift is >= the observed shift
    (i.e. perm overlap <= observed). Add-one smoothed. Crucial for small-n_ko guides.
    """
    if cats is None:
        cats = np.array(sorted({str(c) for c in data.cell_type}), dtype=object)
    ko = np.where(data.perturbation == perturbation)[0]
    ntc = np.where(data.control_pool & ~data.is_perturbed)[0]
    pooled = np.where(data.is_perturbed)[0]

    ko_comp = neighbor_composition(data, ko, adj, cats)
    ntc_comp = neighbor_composition(data, ntc, adj, cats)
    pooled_comp = neighbor_composition(data, pooled, adj, cats)
    ov_ntc = composition_overlap(ntc_comp, ko_comp)
    ov_pooled = composition_overlap(pooled_comp, ko_comp)

    out = dict(guide=str(perturbation), n_ko=int(len(ko)),
               overlap_ntc=ov_ntc, overlap_pooled=ov_pooled,
               delta_ntc=(ko_comp - ntc_comp), delta_pooled=(ko_comp - pooled_comp),
               ko_comp=ko_comp, ntc_comp=ntc_comp, pooled_comp=pooled_comp,
               cats=[str(c) for c in cats])
    if n_perm and len(ko) > 0 and len(ntc) > 0:
        pool = np.concatenate([ko, ntc])
        n_ko = len(ko)
        rng = np.random.default_rng(seed)
        ge = 0
        for _ in range(n_perm):
            perm = rng.permutation(pool)
            fke_ko, fke_nt = perm[:n_ko], perm[n_ko:]
            ov = composition_overlap(neighbor_composition(data, fke_nt, adj, cats),
                                     neighbor_composition(data, fke_ko, adj, cats))
            if ov <= ov_ntc:                       # permuted shift >= observed shift
                ge += 1
        out["p_value"] = (ge + 1) / (n_perm + 1)
    return out


def niche_effect_board(data, edges, min_ko: int = 30, n_perm: int = 1000, seed: int = 0,
                       perturbations=None) -> dict:
    """Descriptive board: per-guide observed niche effect across a dataset, sorted by the
    guide-SPECIFIC shift (lowest `overlap_pooled` first). Runs on any StandardData; guards the
    degenerate single-cell-type case (e.g. Cheng's single line) where composition is meaningless.
    Only guides with >= `min_ko` perturbed cells are scored."""
    cats = np.array(sorted({str(c) for c in data.cell_type}), dtype=object)
    if len(cats) < 2:
        return {"rows": [], "cats": [str(c) for c in cats], "n_guides": 0,
                "note": "degenerate: <2 cell types (composition undefined)"}
    adj = build_adjacency(edges, data.n_cells)
    perts = perturbations if perturbations is not None else data.perturbations()
    rows = []
    for p in perts:
        if int((data.perturbation == p).sum()) < min_ko:
            continue
        rows.append(composition_effect(data, p, adj, cats=cats, n_perm=n_perm, seed=seed))
    rows.sort(key=lambda r: r["overlap_pooled"])
    return {"rows": rows, "cats": [str(c) for c in cats], "n_guides": len(rows)}


def niche_composition(labels, cats) -> np.ndarray:
    """Cell-type labels -> (C,) simplex over `cats` (L1-normalised counts). Empty -> all-zero row."""
    labels = np.asarray(labels, dtype=object)
    cats = np.asarray(cats, dtype=object)
    comp = np.zeros(len(cats), dtype=float)
    if len(labels) == 0:
        return comp
    index = {c: i for i, c in enumerate(cats)}
    for lab in labels:
        j = index.get(lab)
        if j is not None:
            comp[j] += 1.0
    total = comp.sum()
    return comp / total if total > 0 else comp


def composition_overlap(pred_comp, gt_comp) -> float:
    """Overlap = 1 - TV(pred, gt) in [0, 1]. 1 = identical composition. Reuses the comp_l1 (TV)
    metric on single-row stacks so TV has exactly one definition in the codebase."""
    tv = float(get_metric("comp_l1").compute(
        np.asarray(pred_comp, float)[None, :], np.asarray(gt_comp, float)[None, :]))
    return 1.0 - tv


def composition_board(gt_labels, ref_labels, pred_labels=None, cats=None) -> dict:
    """Pure label-space Overlap board. `gt_labels` = observed perturbed-niche labels; `ref_labels`
    = reference (control) niche labels = the 'no-change' baseline; `pred_labels` = {method: labels}.
    Returns Overlap per method (incl 'null' = baseline) and gain = Overlap_method - Overlap_null."""
    pred_labels = dict(pred_labels or {})
    if cats is None:
        seen = list(gt_labels) + list(ref_labels)
        for v in pred_labels.values():
            seen += list(v)
        cats = np.array(sorted({str(s) for s in seen}), dtype=object)
    gt = niche_composition(gt_labels, cats)
    comps = {"null": niche_composition(ref_labels, cats)}
    for name, lab in pred_labels.items():
        comps[name] = niche_composition(lab, cats)
    overlap = {k: composition_overlap(c, gt) for k, c in comps.items()}
    gain = {k: overlap[k] - overlap["null"] for k in comps}
    return {"overlap": overlap, "gain": gain, "gt_comp": gt,
            "cats": [str(c) for c in cats], "n": int(len(gt_labels))}


def composition_eval(data, perturbation, edges, annotator=None, pred_niches=None) -> dict:
    """Bridge: assemble the Overlap board for one perturbation.

    annotator given -> observed/reference/predicted niches are labelled from EXPRESSION by the one
    frozen annotator (gt = annotator(observed), pred = annotator(predicted) -> fair); pred_niches
    is {method: expression_array}. annotator None -> use native data.cell_type for observed/
    reference and treat pred_niches as {method: label_array}."""
    g = propagation_gt(data, perturbation, edges)
    pert_nb, ref_nb = g["pert_nb"], g["ref_nb"]
    pred_niches = pred_niches or {}
    if annotator is not None:
        gn = data.gene_names
        gt_labels = annotator.predict(data.X[pert_nb], gene_names=gn)
        ref_labels = annotator.predict(data.X[ref_nb], gene_names=gn)
        cats = annotator.cats_
        pred_labels = {k: annotator.predict(np.asarray(v, float), gene_names=gn)
                       for k, v in pred_niches.items()}
    else:
        gt_labels = data.cell_type[pert_nb]
        ref_labels = data.cell_type[ref_nb]
        cats = np.array(sorted({str(c) for c in data.cell_type}), dtype=object)
        pred_labels = {k: np.asarray(v, dtype=object) for k, v in pred_niches.items()}
    return composition_board(gt_labels, ref_labels, pred_labels, cats=cats)
