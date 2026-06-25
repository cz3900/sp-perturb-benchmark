"""StandardData -> CPA input (raw-counts binary AnnData). CPA (Lotfollahi et al. 2023) uses an NB
likelihood, so X must be RAW counts; obs['condition'] in {'stimulated','control'}. CPA's
counterfactual prediction lands in obsm['CPA_pred'] (NOT .X) — the offline runner reads it there
and dumps the per-center-aligned seed."""
from .counts_export import export_counts_h5


def export_to_cpa_h5(data, perturbation, counts_X, path):
    return export_counts_h5(data, perturbation, counts_X, path,
                            stim_cond="stimulated", ctrl_cond="control",
                            cell_type_key="cell_type", gene_key="gene_names")
