"""StandardData -> SpatialProp input as a REAL AnnData .h5ad. SpatialProp's API reads it with
sc.read_h5ad and requires obs['celltype'], obs['mouse_id'] (its hard-coded sample-id column),
obsm['spatial'], and RAW counts in X (it normalize_total's internally). mouse_id is set from the
dataset batch; the runner re-splits into >=2 disjoint train/test groups when a slice has only one."""
import numpy as np
import anndata as ad
import pandas as pd


def export_to_spatialprop_h5ad(data, counts_X, path):
    counts_X = np.asarray(counts_X, float)
    if counts_X.shape != (data.n_cells, data.n_genes):
        raise ValueError(f"counts_X shape {counts_X.shape} != ({data.n_cells}, {data.n_genes})")
    genes = [str(g) for g in data.gene_names]
    obs = pd.DataFrame({
        "celltype": [str(c) for c in data.cell_type],
        "mouse_id": [str(b) for b in data.batch],
        "batch": [str(b) for b in data.batch],
    })
    adata = ad.AnnData(X=counts_X, obs=obs,
                       var=pd.DataFrame(index=pd.Index(genes, name=None)))
    adata.obsm["spatial"] = np.asarray(data.coords, float)
    adata.write_h5ad(path)
    return {"n_cells": int(data.n_cells)}
