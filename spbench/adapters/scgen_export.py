"""Export a StandardData (one slice) to scGEN's input format, as a minimal AnnData `.h5ad`.

scGEN (Lotfollahi et al., Nature Methods 2019) consumes an AnnData with:
  X                  (N, G)  LOG-NORMALIZED expression (is_count_data=False)
  obs['condition']   (N,)    'control' | 'stimulated'  -> SCGEN.setup_anndata batch_key
  obs['cell_type']   (N,)    cell type                 -> labels_key
  obs['orig_idx']    (N,)    StandardData cell index of each kept row (loader alignment)
  var['gene_names']  (G,)    panel genes
  uns.attrs          normalization recipe (target_sum, log1p) so the shared-venv eval_X can
                     reproduce the EXACT same log-norm space the offline scgen env trained on.

scGEN models exactly ONE control->stimulated shift, so per perturbation P we keep only that guide's
cells ('stimulated') + control cells ('control'); 'none' (UNLABELED) and all OTHER guides are
dropped. log-norm X is precomputed once via build_lognorm_X (raw integer counts -> normalize_total
1e4 + log1p), NOT from data.X (which is z-scored/raw_scaled).
"""
import numpy as np
import h5py
from ..data import CONTROL

TARGET_SUM = 1e4


def build_lognorm_X(data):
    """Log-normalized (N, G) matrix from raw INTEGER counts in `data.meta['counts']`:
    normalize_total(target_sum=1e4) + log1p, in plain numpy (scanpy not required in the venv).
    Asserts integer counts (mirrors export_to_concert_h5) to fail loud on a pre-normalized layer."""
    if "counts" not in data.meta:
        raise ValueError("data.meta['counts'] missing; load the adapter with the raw counts layer")
    counts = np.asarray(data.meta["counts"], float)
    if counts.shape != (data.n_cells, data.n_genes):
        raise ValueError(f"counts shape {counts.shape} != ({data.n_cells}, {data.n_genes})")
    if not np.allclose(counts, np.round(counts), atol=1e-6):
        raise ValueError("build_lognorm_X requires RAW INTEGER counts, but meta['counts'] is "
                         "non-integer (probably a pre-normalized layer).")
    rowsum = counts.sum(1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return np.log1p(counts / rowsum * TARGET_SUM)


def export_to_scgen_h5(data, perturbation, lognorm_X, path):
    """Write `data`'s binary (control vs `perturbation`) scGEN AnnData to `path`.

    lognorm_X : (n_cells, n_genes) precomputed log-norm matrix (from build_lognorm_X).
    Keeps guide==perturbation cells ('stimulated') + control cells ('control'); drops everything
    else. Returns {'n_stim','n_ctrl'} for a sanity log."""
    lognorm_X = np.asarray(lognorm_X, float)
    if lognorm_X.shape != (data.n_cells, data.n_genes):
        raise ValueError(f"lognorm_X shape {lognorm_X.shape} != ({data.n_cells}, {data.n_genes})")
    mask_stim = data.perturbation == perturbation
    mask_ctrl = data.perturbation == CONTROL
    keep = np.where(mask_stim | mask_ctrl)[0]
    cond = np.where(mask_stim[keep], "stimulated", "control")

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=lognorm_X[keep])
        g_obs = f.create_group("obs")
        g_obs.create_dataset("condition", data=np.asarray(cond, dtype="S"))
        g_obs.create_dataset("cell_type",
                             data=np.asarray([str(c) for c in data.cell_type[keep]], dtype="S"))
        g_obs.create_dataset("orig_idx", data=keep.astype(np.int64))
        g_var = f.create_group("var")
        g_var.create_dataset("gene_names",
                             data=np.asarray([str(g) for g in data.gene_names], dtype="S"))
        g_uns = f.create_group("uns")
        g_uns.attrs["target_sum"] = float(TARGET_SUM)
        g_uns.attrs["log1p"] = 1

    return {"n_stim": int(mask_stim.sum()), "n_ctrl": int(mask_ctrl.sum())}
