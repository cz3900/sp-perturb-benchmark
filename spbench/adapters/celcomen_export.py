"""StandardData -> Celcomen input. Celcomen (CCE inference + Simcomen counterfactual) needs RAW
integer counts (no normalization), obs['cell_type'], obs['batch'], obsm['spatial'] (it builds the
k-hop spatial graph internally), and gene names. KO is applied in Simcomen via set_sphex on the
in-panel guide gene's column."""
import numpy as np, h5py


def export_to_celcomen_h5(data, counts_X, path):
    counts_X = np.asarray(counts_X, float)
    if counts_X.shape != (data.n_cells, data.n_genes):
        raise ValueError(f"counts_X shape {counts_X.shape} != ({data.n_cells}, {data.n_genes})")
    if not np.allclose(counts_X, np.round(counts_X), atol=1e-6):
        raise ValueError("Celcomen requires RAW INTEGER counts; pass the raw count layer.")
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=counts_X)
        g = f.create_group("obs")
        g.create_dataset("cell_type", data=np.asarray([str(c) for c in data.cell_type], dtype="S"))
        g.create_dataset("batch", data=np.asarray([str(b) for b in data.batch], dtype="S"))
        f.create_group("obsm").create_dataset("spatial", data=np.asarray(data.coords, float))
        f.create_group("var").create_dataset("gene_names",
                        data=np.asarray([str(x) for x in data.gene_names], dtype="S"))
    return {"n_cells": int(data.n_cells)}
