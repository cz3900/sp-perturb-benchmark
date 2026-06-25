"""Offline CPA runner — runs in the dedicated `cpa` conda env (py<3.11), NOT the shared venv.

Pipeline per perturbation P (dump contract mirrors scripts/scgen):
  1. (venv) export `{P}.h5ad` via spbench.adapters.cpa_export.export_to_cpa_h5 (raw counts,
     obs['condition'] stimulated/control) + `{P}_centers.npz` ('center_idx','cell_type').
  2. (cpa env) THIS script: train CPA, predict the counterfactual for the CONTROL cells, read the
     prediction from obsm['CPA_pred'], average per cell type -> profile bank, map each center to its
     cell type (aggregate-control contract), dump `{P}_seed.h5ad` aligned to centers order.
  3. (venv) SeedDumpModel('cpa', {P: ...}) serves it.

Pure functions below are import-safe in the venv (tested there); cpa/scvi imported lazily in main."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/ on path
from _dump_io import aggregate_control_predict, write_seed_dump  # noqa: E402


def celltype_profiles_from_pred(pred, cell_types):
    """Per-cell-type mean of CPA's counterfactual predictions (obsm['CPA_pred'] rows), key None =
    global mean fallback. pred (m, G), cell_types (m,)."""
    pred = np.asarray(pred, float); cell_types = np.asarray(cell_types)
    prof = {None: pred.mean(0)}
    for ct in np.unique(cell_types):
        prof[ct] = pred[cell_types == ct].mean(0)
    return prof


def build_seed(center_cell_types, profiles):
    """Aggregate-control seed_pred (n_centers, G) — thin wrapper over the shared helper."""
    return aggregate_control_predict(center_cell_types, profiles)


def main():                                              # pragma: no cover (needs the cpa env)
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True); ap.add_argument("--centers", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import cpa, scanpy as sc                              # lazy: only in the cpa env
    adata = sc.read_h5ad(args.h5ad)

    # Train CPA on the full AnnData (control + stimulated), then predict the counterfactual for the
    # control cells. See ENV_SETUP.md for the exact setup_anndata / CPA / model.predict call sequence.
    cpa.CPA.setup_anndata(
        adata, perturbation_key="condition", control_group="control",
        dosage_key=None, categorical_covariate_keys=["cell_type"], is_count_data=True,
    )
    model = cpa.CPA(adata)
    model.train(max_epochs=200, early_stopping=True)

    # Counterfactual: shift the CONTROL cells to 'stimulated' and read the prediction. CPA writes the
    # counterfactual expression into obsm['CPA_pred'] (NOT .X — load-bearing gotcha, see ENV_SETUP.md).
    adata_ctrl = adata[adata.obs["condition"] == "control"].copy()
    adata_ctrl.obs["condition"] = "stimulated"
    model.predict(adata_ctrl)
    pred = adata_ctrl.obsm["CPA_pred"]
    ctrl_cell_types = adata_ctrl.obs["cell_type"].values

    npz = np.load(args.centers, allow_pickle=True)
    profiles = celltype_profiles_from_pred(pred, ctrl_cell_types)
    seed = build_seed(npz["cell_type"], profiles)
    write_seed_dump(args.out, seed, npz["center_idx"])
    print(f"wrote {args.out}: seed {seed.shape}, {len(profiles) - 1} cell-type profiles")


if __name__ == "__main__":                               # pragma: no cover
    main()
