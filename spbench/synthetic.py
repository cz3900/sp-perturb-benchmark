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
