# /Users/cz/Documents/ZengLab/model/sp-perturb-benchmark/spbench/reference_aggregate.py
"""Sample-level aggregate control reference (G1).

A per-cell-type aggregate computed from CONTROL cells only (this replaced an earlier feature-space
nearest-neighbour matched-control approach, retired to remove its leakage):

  - expr        : mean expression of control cells of that cell type (global control mean fallback)
  - niche_comp  : mean bystander-neighbour cell-type composition (a simplex over `cell_types`)
  - niche_expr  : mean bystander-neighbour expression

The effect of a perturbation is then (perturbed aggregate - control aggregate), which removes the
matched-control contamination/leakage of feature-space pairing. niche here uses the existing kNN
spatial graph (`graph.neighbors_of`); the squidpy-based niche definition is G2.
"""
from dataclasses import dataclass
import numpy as np

from .graph import neighbors_of


@dataclass
class AggregateControl:
    cell_types: np.ndarray          # sorted unique cell types (the composition axis order)
    expr: dict                      # ct -> (n_genes,) mean control expression
    niche_comp: dict                # ct -> (len(cell_types),) simplex of bystander composition
    niche_expr: dict                # ct -> (n_genes,) mean bystander expression


def _bystanders(data, center, edges):
    """Non-perturbed neighbours of `center` (same semantics as harness._bystanders)."""
    nb = neighbors_of(center, edges)
    return nb[~data.is_perturbed[nb]]


def aggregate_control(data, edges, min_control: int = 1) -> AggregateControl:
    """Per-cell-type aggregate control reference. CONTROL cells only; global control mean as
    the fallback when a cell type has fewer than `min_control` control cells."""
    cell_types = np.unique(data.cell_type)            # already sorted
    ct_index = {ct: i for i, ct in enumerate(cell_types)}
    ctrl = data.control_pool
    gmean = data.X[ctrl].mean(0) if ctrl.any() else data.X.mean(0)

    expr, niche_comp, niche_expr = {}, {}, {}
    for ct in cell_types:
        ctrl_ct = np.where(ctrl & (data.cell_type == ct))[0]
        # (1) mean control expression, global fallback
        if len(ctrl_ct) >= min_control:
            expr[ct] = data.X[ctrl_ct].mean(0).astype(float)
        else:
            expr[ct] = gmean.astype(float)
            ctrl_ct = np.where(ctrl)[0]               # use all control cells for this type's niche too

        # (2)+(3) aggregate bystander niche over the control cells of this type
        comp = np.zeros(len(cell_types), float)
        expr_sum = np.zeros(data.n_genes, float)
        n_nb = 0
        for c in ctrl_ct:
            nb = _bystanders(data, c, edges)
            if len(nb) == 0:
                continue
            for j in nb:
                comp[ct_index[data.cell_type[j]]] += 1.0
            expr_sum += data.X[nb].sum(0)
            n_nb += len(nb)
        if n_nb > 0:
            niche_comp[ct] = comp / comp.sum()
            niche_expr[ct] = expr_sum / n_nb
        else:
            # no bystander neighbours anywhere -> uniform composition, global-mean niche expr
            niche_comp[ct] = np.full(len(cell_types), 1.0 / len(cell_types))
            niche_expr[ct] = gmean.astype(float)

    return AggregateControl(
        cell_types=cell_types, expr=expr, niche_comp=niche_comp, niche_expr=niche_expr,
    )


def control_reference_centers(data, centers):
    """Aggregate-control reference centers (sample-level, no feature-space matching).

    For each perturbed center, return ALL control cells of the SAME cell type (sample-level), with
    NO expression nearest-neighbour matching — so there is no matched-control feature-space leakage
    (the whole point of the aggregate-control reference: the control is the sample's average
    unperturbed cell of that type, not the control cell that happens to look most like the
    perturbed one). Falls back to all control cells when a cell type has no controls. Returns a
    list aligned to `centers`, consumed by harness / propagation_gt directly (same-type centers
    share one array — cheap, read-only).
    """
    ctrl_idx = np.where(data.control_pool)[0]
    by_type, out = {}, []
    for c in centers:
        ct = data.cell_type[c]
        if ct not in by_type:
            same = ctrl_idx[data.cell_type[ctrl_idx] == ct]
            by_type[ct] = same if len(same) else ctrl_idx
        out.append(by_type[ct])
    return out
