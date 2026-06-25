"""Offline SpatialProp runner — dedicated `spatialprop` conda env (torch 2.6 / PyG 2.6), NOT venv.

Pipeline: (venv) export the slice via spatialprop_export; (spatialprop env) THIS script:
train_perturbation_model -> create_perturbation_input_matrix (per-celltype gene multiplier, ONLY
for in-panel guide genes) -> predict_perturbation_effects, dump layers['predicted_tempered'] per P.
No pre-trained weights ship — training is from scratch. Off-panel guide genes cannot be a multiplier
input (same off-panel limit as scGEN). Pure functions tested in the venv."""
import argparse
import numpy as np


def build_multiplier_matrix(guide_gene, genes, celltypes, value=0.0):
    """Per-celltype gene-multiplier input for one guide. Rows = celltypes, cols = genes; the guide
    gene's column = `value` (e.g. 0 for knockout), all others 1.0. Returns None when the guide gene
    is OFF-PANEL (not in `genes`) — it cannot be expressed as a multiplier input."""
    genes = list(genes)
    if guide_gene not in genes:
        return None
    mult = np.ones((len(celltypes), len(genes)), float)
    mult[:, genes.index(guide_gene)] = value
    return mult


def main():                                              # pragma: no cover (needs the spatialprop env)
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True); ap.add_argument("--pert", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import scanpy as sc                                   # lazy
    from spatialprop import (train_perturbation_model, create_perturbation_input_matrix,
                             predict_perturbation_effects)
    adata = sc.read_h5ad(args.h5ad)
    genes = list(adata.var["gene_names"].astype(str))
    celltypes = list(np.unique(adata.obs["celltype"].astype(str)))
    mult = build_multiplier_matrix(args.pert, genes, celltypes)
    if mult is None:
        raise SystemExit(f"guide {args.pert!r} is off-panel; cannot build a multiplier input")
    model = train_perturbation_model(adata)
    inp = create_perturbation_input_matrix(adata, mult)
    pred = predict_perturbation_effects(model, inp)
    adata.layers["predicted_tempered"] = pred
    adata.write_h5ad(args.out)
    print(f"wrote {args.out}: predicted_tempered {pred.shape}")


if __name__ == "__main__":                               # pragma: no cover
    main()
