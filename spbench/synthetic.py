import numpy as np
from .data import StandardData, CONTROL, UNLABELED

def make_synthetic(seed=0, grid=24, n_genes=20, n_perturb=3, prop_radius=2.5):
    """Cells on a grid. A few perturbed centers carry a planted seed shift on one gene
    and a planted propagation shift on a *different* gene for their spatial neighbors.
    Most cells are 'none' (unlabeled background); some are 'control'. Two slices."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in ("s0", "s1"):
        xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
        coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
        n = coords.shape[0]
        X = rng.normal(0, 1, size=(n, n_genes))
        ct = rng.choice(["Hep", "Endo", "T"], size=n, p=[0.7, 0.2, 0.1])
        pert = np.full(n, UNLABELED, dtype=object)
        rows.append([coords, X, ct, pert, b])

    planted = {}
    for pi in range(n_perturb):
        name = f"P{pi}"
        sg, pg = pi, (pi + n_genes // 2) % n_genes   # seed gene, propagation gene (distinct)
        planted[name] = {"seed_gene": sg, "prop_gene": pg}
        for coords, X, ct, pert, b in rows:
            n = coords.shape[0]
            centers = rng.choice(n, size=8, replace=False)
            for c in centers:
                pert[c] = name
                X[c, sg] += 3.0                                  # planted SEED shift
                dist = np.linalg.norm(coords - coords[c], axis=1)
                neigh = (dist > 0) & (dist <= prop_radius)
                X[neigh, pg] += 1.2                              # planted PROPAGATION shift
    for coords, X, ct, pert, b in rows:
        free = np.where(pert == UNLABELED)[0]
        ctrl = rng.choice(free, size=40, replace=False)
        pert[ctrl] = CONTROL

    coords = np.vstack([r[0] for r in rows])
    X = np.vstack([r[1] for r in rows])
    ct = np.concatenate([r[2] for r in rows])
    pert = np.concatenate([r[3] for r in rows]).astype(str)
    batch = np.concatenate([np.full(r[0].shape[0], r[4]) for r in rows])
    return StandardData(
        X=X, coords=coords, perturbation=pert, cell_type=ct, batch=batch,
        gene_names=[f"g{i}" for i in range(n_genes)],
        meta={"name": "synthetic", "planted": planted, "prop_radius": prop_radius},
    )


def make_synthetic_with_effects(
    seed=0, grid=24, n_genes=20, n_centers=8, prop_radius=2.5,
    d1_shift=3.0, d2_shift=1.2, d3_extra=0.6,
):
    """L2 synthetic with *known* D1/D2/D3 effects and known significant/inert labels.

    Builds two grid slices of background cells, then injects three perturbations:
      - 'SIG' (significant): a planted D1 seed shift (one gene on the perturbed cells),
        a planted D2 propagation shift (a distinct gene on their spatial bystander
        neighbors), and a planted D3 composition enrichment (a chosen cell type is
        over-represented among the neighbors of perturbed centers).
      - 'SIG2' (significant): same structure, different genes/cell type, so recovery is
        not a single-perturbation fluke.
      - 'INERT' (negative control): perturbed *labels* are placed, but NO D1/D2/D3 shift
        is injected -> the pipeline must judge it as having no effect.

    meta['effects'] = {
        'significant': [names...], 'inert': [names...],
        'spec': {name: {'d1_gene','d1_shift','d2_gene','d2_shift',
                        'd3_cell_type','d3_extra'}}
    }
    so L2 tests can assert: injected effects are recovered, un-injected dims ~= 0,
    and the inert perturbation is judged effect-free.
    """
    rng = np.random.default_rng(seed)
    cell_types = ["Hep", "Endo", "T"]

    rows = []
    for b in ("s0", "s1"):
        xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
        coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
        n = coords.shape[0]
        X = rng.normal(0, 1, size=(n, n_genes))
        ct = rng.choice(cell_types, size=n, p=[0.7, 0.2, 0.1])
        pert = np.full(n, UNLABELED, dtype=object)
        rows.append([coords, X, ct, pert, b])

    # perturbation specs: two significant (full effect), one inert (zero effect)
    spec = {
        "SIG":   {"d1_gene": 0, "d1_shift": d1_shift,
                  "d2_gene": n_genes // 2, "d2_shift": d2_shift,
                  "d3_cell_type": "Endo", "d3_extra": d3_extra},
        "SIG2":  {"d1_gene": 1, "d1_shift": d1_shift,
                  "d2_gene": n_genes // 2 + 1, "d2_shift": d2_shift,
                  "d3_cell_type": "T", "d3_extra": d3_extra},
        "INERT": {"d1_gene": 2, "d1_shift": 0.0,
                  "d2_gene": n_genes // 2 + 2, "d2_shift": 0.0,
                  "d3_cell_type": "Endo", "d3_extra": 0.0},
    }

    for name, s in spec.items():
        sg, pg = s["d1_gene"], s["d2_gene"]
        for coords, X, ct, pert, b in rows:
            n = coords.shape[0]
            free = np.where(pert == UNLABELED)[0]
            centers = rng.choice(free, size=n_centers, replace=False)
            for c in centers:
                pert[c] = name
                X[c, sg] += s["d1_shift"]                          # D1 seed shift
                dist = np.linalg.norm(coords - coords[c], axis=1)
                neigh = (dist > 0) & (dist <= prop_radius)
                X[neigh, pg] += s["d2_shift"]                      # D2 propagation shift
                if s["d3_extra"] > 0:                              # D3 composition enrichment
                    flip = np.where(neigh & (pert == UNLABELED))[0]
                    take = rng.random(len(flip)) < s["d3_extra"]
                    ct[flip[take]] = s["d3_cell_type"]

    # remaining free cells become controls
    for coords, X, ct, pert, b in rows:
        free = np.where(pert == UNLABELED)[0]
        k = min(40, len(free))
        ctrl = rng.choice(free, size=k, replace=False)
        pert[ctrl] = CONTROL

    coords = np.vstack([r[0] for r in rows])
    X = np.vstack([r[1] for r in rows])
    ct = np.concatenate([r[2] for r in rows])
    pert = np.concatenate([r[3] for r in rows]).astype(str)
    batch = np.concatenate([np.full(r[0].shape[0], r[4]) for r in rows])
    return StandardData(
        X=X, coords=coords, perturbation=pert, cell_type=ct, batch=batch,
        gene_names=[f"g{i}" for i in range(n_genes)],
        meta={"name": "synthetic_effects", "prop_radius": prop_radius,
              "effects": {"significant": ["SIG", "SIG2"],
                          "inert": ["INERT"], "spec": spec}},
    )
