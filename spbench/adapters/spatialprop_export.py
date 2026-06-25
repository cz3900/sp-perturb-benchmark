"""StandardData -> SpatialProp input. SpatialProp expects a single AnnData with obs['mouse_id']
(batch), obs['celltype'] (must be a value seen in training, else silently dropped), obsm['spatial']
(coords), and RAW counts in X (it does normalize_total + per-celltype gene multiplier internally).
Unlike the seed adapters this exports the WHOLE slice (not a binary control/perturbation split) —
SpatialProp models the spatial graph, so all cells are needed."""
import numpy as np, h5py


def export_to_spatialprop_h5(data, counts_X, path):
    counts_X = np.asarray(counts_X, float)
    if counts_X.shape != (data.n_cells, data.n_genes):
        raise ValueError(f"counts_X shape {counts_X.shape} != ({data.n_cells}, {data.n_genes})")
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=counts_X)
        g_obs = f.create_group("obs")
        g_obs.create_dataset("mouse_id", data=np.asarray([str(b) for b in data.batch], dtype="S"))
        g_obs.create_dataset("celltype",
                             data=np.asarray([str(c) for c in data.cell_type], dtype="S"))
        f.create_group("obsm").create_dataset("spatial", data=np.asarray(data.coords, float))
        f.create_group("var").create_dataset("gene_names",
                             data=np.asarray([str(g) for g in data.gene_names], dtype="S"))
    return {"n_cells": int(data.n_cells)}
