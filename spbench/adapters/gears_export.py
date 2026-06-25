# spbench/adapters/gears_export.py
"""StandardData -> GEARS input. GEARS (Roohani et al. 2024) expects obs['condition'] = 'GENE+ctrl'
(perturbed) / 'ctrl' (control) and a var['gene_name'] (singular) column; X is raw counts. GEARS
predicts ONE mean expression vector per perturbation (not per cell), which the offline runner
broadcasts to the n_centers via scripts/gears/run_gears.broadcast_seed."""
from .counts_export import export_counts_h5


def export_to_gears_h5(data, perturbation, counts_X, path):
    return export_counts_h5(data, perturbation, counts_X, path,
                            stim_cond=f"{perturbation}+ctrl", ctrl_cond="ctrl",
                            cell_type_key="cell_type", gene_key="gene_name")
