# scripts/biolord/run_biolord.py
"""Offline biolord runner — dedicated `biolord` conda env (scvi-tools + jax), NOT the shared venv.

Per perturbation P: export `{P}.h5ad` (biolord_export) + `{P}_centers.npz`; (biolord env) train,
build the counterfactual by OVERRIDING the condition tensor-dict to 'stimulated' for control cells
and decoding (NOT the default forward), average per cell type -> profile bank, dump `{P}_seed.h5ad`.
Pure functions tested in the venv."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _dump_io import aggregate_control_predict, write_seed_dump  # noqa: E402


def build_seed(center_cell_types, profiles):
    """Aggregate-control seed_pred (n_centers, G)."""
    return aggregate_control_predict(center_cell_types, profiles)


def main():                                              # pragma: no cover (needs the biolord env)
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True); ap.add_argument("--centers", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import biolord, scanpy as sc                          # lazy
    adata = sc.read_h5ad(args.h5ad)

    # Train biolord on the full AnnData (control + stimulated). 'condition' is the disentangled
    # ordered/categorical attribute; 'cell_type' is a known covariate. See ENV_SETUP.md for the
    # exact Biolord.setup_anndata / Biolord / train call sequence.
    biolord.Biolord.setup_anndata(adata, categorical_attributes_keys=["condition", "cell_type"])
    model = biolord.Biolord(adata, n_latent=32)
    model.train(max_epochs=200, early_stopping=True)

    # Counterfactual: for the CONTROL cells, OVERRIDE the condition tensor-dict to 'stimulated' and
    # DECODE the perturbed state. The default forward is reconstruction (no-effect seed) — see
    # ENV_SETUP.md for the tensor-dict override recipe that yields pred (n_ctrl, G).
    adata_ctrl = adata[adata.obs["condition"] == "control"].copy()
    pred = model.compute_prediction_adata(  # noqa: F841  (signature per ENV_SETUP.md override)
        adata_ctrl, target_attributes={"condition": "stimulated"})
    ctrl_cell_types = adata_ctrl.obs["cell_type"].values
    profiles = {None: np.asarray(pred, float).mean(0)}    # noqa: F821
    for ct in np.unique(ctrl_cell_types):
        profiles[ct] = np.asarray(pred, float)[ctrl_cell_types == ct].mean(0)

    npz = np.load(args.centers, allow_pickle=True)
    seed = build_seed(npz["cell_type"], profiles)
    write_seed_dump(args.out, seed, npz["center_idx"])
    print(f"wrote {args.out}: seed {seed.shape}, {len(profiles) - 1} cell-type profiles")


if __name__ == "__main__":                               # pragma: no cover
    main()
