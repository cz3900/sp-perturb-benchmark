"""Offline scGEN runner — runs in the dedicated `scgen` conda env (NOT the shared .venv).

Pipeline per perturbation P (file-dump contract, mirrors scripts/concert):
  1. (shared venv) export `{P}.h5ad` via spbench.adapters.scgen_export.export_to_scgen_h5
     (log-norm X, obs['condition'] control/stimulated, obs['cell_type'], obs['orig_idx']) and a
     `{P}_centers.npz` with arrays 'center_idx' (int, the StandardData centers the harness scores)
     and 'cell_type' (str, each center's cell type), both in centers order.
  2. (scgen env) THIS script: train one scGEN, decode the latent delta on the CELL-TYPE-MEAN control
     profile (aggregate-control contract — NOT per matched cell), build a per-cell-type predicted
     profile bank, map each center to its cell type, and dump the FINAL seed_pred (n_centers x G)
     aligned to centers order to `{P}_seed.h5ad` (/X + /obs/center_idx) for ScgenSeedModel.
  3. (shared venv) ScgenSeedModel reads `{P}_seed.h5ad` and serves it as the model seed.

Design Part C fixes honored: (3) aggregate-control = decode(delta + latent(mean_ctrl_of_celltype)),
collapsing N predictions to a handful of cell-type profiles; (2) the loader consumes the cached
aligned array directly, so no per-rc re-derivation and no ref_idx third arg.

The numpy/h5py helpers below are import-safe in the shared venv (tested there); scgen/scvi/anndata
are imported lazily inside train_celltype_profiles / main so importing this module never requires
them.
"""
import argparse
import numpy as np
import h5py


def aggregate_control_predict(center_cell_types, ct_profiles):
    """Map each center to its cell type's aggregate predicted profile.

    center_cell_types : (n_centers,) cell-type label per center (centers order).
    ct_profiles       : {cell_type: (G,) predicted perturbed profile}; key None = global fallback.
    Returns seed_pred (n_centers, G) aligned to centers order — the aggregate-control prediction."""
    center_cell_types = np.asarray(center_cell_types)
    glob = ct_profiles.get(None)
    G = len(next(iter(ct_profiles.values())))
    out = np.empty((len(center_cell_types), G), float)
    for i, ct in enumerate(center_cell_types):
        prof = ct_profiles.get(ct, glob)
        if prof is None:
            raise KeyError(f"no profile for cell type {ct!r} and no global (None) fallback")
        out[i] = prof
    return out


def write_seed_dump(path, seed_pred, center_idx):
    """Write the final aligned seed_pred + center indices for ScgenSeedModel to read.
    /X = (n_centers, G); /obs/center_idx = (n_centers,) StandardData center indices."""
    seed_pred = np.asarray(seed_pred, float)
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=seed_pred)
        g_obs = f.create_group("obs")
        g_obs.create_dataset("center_idx", data=np.asarray(center_idx, np.int64))


def train_celltype_profiles(in_h5ad, epochs):
    """Train one scGEN on `in_h5ad` and return {cell_type: (G,) predicted perturbed profile, plus
    None: global mean}, by decoding the latent delta on the CELL-TYPE-MEAN control profile.

    Lazy-imports scgen/scvi/anndata — only runs in the scgen conda env."""
    import anndata as ad
    from scgen import SCGEN

    with h5py.File(in_h5ad, "r") as f:
        X = np.asarray(f["X"], float)
        cond = np.array(f["obs"]["condition"]).astype(str)
        cell_type = np.array(f["obs"]["cell_type"]).astype(str)
        gene_names = np.array(f["var"]["gene_names"]).astype(str)
    adata = ad.AnnData(X=X)
    adata.obs["condition"] = cond
    adata.obs["cell_type"] = cell_type
    adata.var_names = gene_names

    SCGEN.setup_anndata(adata, batch_key="condition", labels_key="cell_type")
    model = SCGEN(adata, n_hidden=800, n_latent=100, n_layers=2, dropout_rate=0.2)
    model.train(max_epochs=epochs, batch_size=32, early_stopping=True,
                early_stopping_patience=25)

    profiles = {}
    uniq_ct = np.unique(cell_type)
    for ct in uniq_ct:
        # scGEN.predict shifts control cells of this cell type to 'stimulated'; the aggregate-control
        # contract = take the MEAN over that cell type's predicted-perturbed cells (one profile/ct).
        pred, _ = model.predict(ctrl_key="control", stim_key="stimulated",
                                celltype_to_predict=ct)
        profiles[ct] = np.asarray(pred.X, float).mean(0)
    profiles[None] = np.mean([profiles[ct] for ct in uniq_ct], 0)   # global fallback
    return profiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_h5ad", required=True, help="{P}.h5ad from export_to_scgen_h5")
    ap.add_argument("--centers", required=True,
                    help="{P}_centers.npz with 'center_idx' (int) and 'cell_type' (str), centers order")
    ap.add_argument("--out", required=True, help="{P}_seed.h5ad for ScgenSeedModel")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    c = np.load(args.centers, allow_pickle=True)
    center_idx = np.asarray(c["center_idx"], np.int64)
    center_cell_types = np.asarray(c["cell_type"]).astype(str)

    profiles = train_celltype_profiles(args.in_h5ad, args.epochs)
    seed_pred = aggregate_control_predict(center_cell_types, profiles)
    write_seed_dump(args.out, seed_pred, center_idx)
    print(f"wrote {args.out}: seed_pred {seed_pred.shape}, {len(profiles) - 1} cell-type profiles")


if __name__ == "__main__":
    main()
