"""Map a model prediction over its own gene subset onto the benchmark panel column order, filling
genes the model did not predict from a fallback (the unperturbed input = 'no change'). Keeps the
loader/scoring on a single full-panel gene axis regardless of which genes the model modelled."""
import numpy as np


def align_prediction_to_panel(pred, pred_genes, panel_genes, fallback):
    """pred: (N, len(pred_genes)). fallback: (N, len(panel_genes)) unperturbed input in panel order.
    Returns (N, len(panel_genes)): fallback with each predicted gene's column overwritten by pred."""
    pred = np.asarray(pred, float); fallback = np.asarray(fallback, float)
    panel_genes = list(map(str, panel_genes)); pred_genes = list(map(str, pred_genes))
    if fallback.shape[1] != len(panel_genes):
        raise ValueError(f"fallback has {fallback.shape[1]} cols != {len(panel_genes)} panel genes")
    if pred.shape[0] != fallback.shape[0]:
        raise ValueError(f"pred has {pred.shape[0]} rows != fallback {fallback.shape[0]}")
    out = fallback.copy()
    idx = {g: j for j, g in enumerate(panel_genes)}
    for j, g in enumerate(pred_genes):
        if g in idx:
            out[:, idx[g]] = pred[:, j]
    return out
