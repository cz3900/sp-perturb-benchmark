# spbench/adapters/biolord_export.py
"""StandardData -> biolord input (raw-counts binary AnnData, NB decoder). biolord (Piran et al.
2024) disentangles condition; its counterfactual is NOT the default forward — the offline runner
overrides the condition tensor-dict to decode the perturbed state for control cells."""
from .counts_export import export_counts_h5


def export_to_biolord_h5(data, perturbation, counts_X, path):
    return export_counts_h5(data, perturbation, counts_X, path,
                            stim_cond="stimulated", ctrl_cond="control",
                            cell_type_key="cell_type", gene_key="gene_names")
