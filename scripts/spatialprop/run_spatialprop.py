"""Offline SpatialProp runner — dedicated `spatialprop` env (torch2.6/PyG2.6 + the spatial_gnn pkg).
Real API: train_perturbation_model (raw-file mode) -> create_perturbation_input_matrix (KO = gene
multiplier 0 across all cell types) -> predict_perturbation_effects. The input .h5ad needs
obs['mouse_id'] with >=2 disjoint groups for the train/test split; for a single-slice dataset we
synthesize a deterministic 2-group split. Predicts the whole slice (use_ids=None); NaN (unpredicted)
cells are filled with the normalized input (no change). Output dumped to layers['predicted_tempered']
for the loader. Pure helpers are unit-tested; main() needs the env + GPU."""
import argparse
import numpy as np


def build_perturbation_dict(guide_gene, celltypes, genes, value=0.0):
    """{celltype: {guide_gene: value}} for create_perturbation_input_matrix. Returns None when the
    guide is OFF-PANEL (not in `genes`) — it cannot be expressed as a multiplier (option 2)."""
    if str(guide_gene) not in set(map(str, genes)):
        return None
    return {str(ct): {str(guide_gene): float(value)} for ct in celltypes}


def two_group_split(n, seed=0):
    """Deterministic 50/50 assignment of n cells into mouse_id groups 'g0'/'g1' so SpatialProp's
    hard-coded train/test id split has two disjoint groups even for a single-slice dataset."""
    rng = np.random.default_rng(seed)
    g = np.where(rng.random(n) < 0.5, "g0", "g1")
    if (g == "g0").all() or (g == "g1").all():     # guard tiny n
        g = np.array(["g0" if i % 2 == 0 else "g1" for i in range(n)])
    return g


def main():                                          # pragma: no cover (needs spatialprop env + GPU)
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)         # real AnnData from export_to_spatialprop_h5ad
    ap.add_argument("--pert", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    import torch, scanpy as sc
    from spatial_gnn.api.perturbation_api import (train_perturbation_model,
        create_perturbation_input_matrix, predict_perturbation_effects)

    adata = sc.read_h5ad(args.h5ad)
    genes = list(adata.var_names.astype(str))
    celltypes = list(adata.obs["celltype"].astype(str).unique())
    pert = build_perturbation_dict(args.pert, celltypes, genes)
    if pert is None:
        raise SystemExit(f"guide {args.pert!r} is off-panel; cannot perturb")
    # ensure >=2 disjoint mouse_id groups for the train/test split
    if adata.obs["mouse_id"].astype(str).nunique() < 2:
        adata.obs["mouse_id"] = two_group_split(adata.n_obs)
        adata.write_h5ad(args.h5ad)                  # persist split for the API's file reads
    ids = sorted(adata.obs["mouse_id"].astype(str).unique().tolist())   # disjoint id groups
    exp = f"sp_{args.pert}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, gene_names, (_, _, trained_model_path) = train_perturbation_model(
        file_path=args.h5ad, train_ids=ids[:-1], test_ids=ids[-1:], exp_name=exp,
        k_hop=2, augment_hop=2, center_celltypes="all", node_feature="expression",
        inject_feature="none", learning_rate=1e-4, loss="weightedl1", epochs=args.epochs,
        num_cells_per_ct_id=100, normalize_total=True, predict_celltype=False, pool="center",
        device=device)
    save_path = args.h5ad.replace(".h5ad", f"_{args.pert}_perturbed.h5ad")
    create_perturbation_input_matrix(adata, pert, mask_key="perturbed_input",
                                     save_path=save_path, normalize_total=True)
    result = predict_perturbation_effects(adata_path=save_path, model_path=trained_model_path,
                                          exp_name=exp, use_ids=ids, whole_tissue=True)
    pred = np.asarray(result.layers["predicted_tempered"], float)
    # fill NaN (unpredicted) cells with the normalized input (no change)
    base = np.asarray(result.layers.get("predicted_unperturbed", result.X), float)
    nan = ~np.isfinite(pred)
    pred[nan] = base[nan]
    import h5py
    with h5py.File(args.out, "w") as f:
        f.create_dataset("X", data=np.zeros_like(pred))
        f.create_group("layers").create_dataset("predicted_tempered", data=pred)
    print(f"wrote {args.out}: predicted_tempered {pred.shape}")


if __name__ == "__main__":                           # pragma: no cover
    main()
