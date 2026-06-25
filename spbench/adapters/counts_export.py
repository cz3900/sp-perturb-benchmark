"""Export a StandardData (one slice) to a raw-counts, binary (control vs one perturbation) AnnData
`.h5ad` for the count-based seed models (CPA / GEARS / biolord, all NB-likelihood). Mirrors
scgen_export.export_to_scgen_h5 but keeps RAW INTEGER counts in X (the models normalize internally)
and lets each model name its own obs['condition'] values, cell-type obs key, and var gene key."""
import numpy as np, h5py
from ..data import CONTROL


def build_counts_X(data):
    """Raw integer (N, G) counts from data.meta['counts'], with the same integer guard as
    scgen_export.build_lognorm_X (fail loud on a pre-normalized layer)."""
    if "counts" not in data.meta:
        raise ValueError("data.meta['counts'] missing; load the adapter with the raw counts layer")
    counts = np.asarray(data.meta["counts"], float)
    if counts.shape != (data.n_cells, data.n_genes):
        raise ValueError(f"counts shape {counts.shape} != ({data.n_cells}, {data.n_genes})")
    if not np.allclose(counts, np.round(counts), atol=1e-6):
        raise ValueError("build_counts_X requires RAW INTEGER counts (got a non-integer layer).")
    return counts


def export_counts_h5(data, perturbation, counts_X, path, *, stim_cond, ctrl_cond,
                     cell_type_key="cell_type", gene_key="gene_names"):
    """Write the binary (control vs `perturbation`) raw-counts AnnData to `path`.
    stim_cond / ctrl_cond : obs['condition'] string for perturbed / control rows.
    Returns {'n_stim','n_ctrl'}."""
    counts_X = np.asarray(counts_X, float)
    if counts_X.shape != (data.n_cells, data.n_genes):
        raise ValueError(f"counts_X shape {counts_X.shape} != ({data.n_cells}, {data.n_genes})")
    mask_stim = data.perturbation == perturbation
    mask_ctrl = data.perturbation == CONTROL
    keep = np.where(mask_stim | mask_ctrl)[0]
    cond = np.where(mask_stim[keep], stim_cond, ctrl_cond)
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=counts_X[keep])
        g_obs = f.create_group("obs")
        g_obs.create_dataset("condition", data=np.asarray(cond, dtype="S"))
        g_obs.create_dataset(cell_type_key,
                             data=np.asarray([str(c) for c in data.cell_type[keep]], dtype="S"))
        g_obs.create_dataset("orig_idx", data=keep.astype(np.int64))
        g_var = f.create_group("var")
        g_var.create_dataset(gene_key,
                             data=np.asarray([str(g) for g in data.gene_names], dtype="S"))
    return {"n_stim": int(mask_stim.sum()), "n_ctrl": int(mask_ctrl.sum())}
