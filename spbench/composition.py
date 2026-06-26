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

from .metrics import get_metric
from .propagation_gt import propagation_gt

DEPLOYABLE = "model+learned"   # method label for the deployed model's predicted niche (matches compare.DEPLOYABLE)


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
