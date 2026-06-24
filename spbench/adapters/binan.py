import numpy as np
from .base import DatasetAdapter
from ..data import StandardData


def _assemble_binan(X, coords, pert_onehot, gene_names, tumor_full_idx_1based):
    """Assemble the Binan tumors all-cells StandardData (pure; the file I/O lives in the adapter).
    pert_onehot: (n_cells, n_guides) 0/1 — perturbation is 'control' (no guide, sum 0), 'guide_<k>'
    (single guide, sum 1; k = argmax), or 'none' (multiplet, sum>=2). cell_type: 'tumor' for cells
    whose 1-based full-cell index is in `tumor_full_idx_1based`, else 'other'."""
    X = np.asarray(X, float)
    n = X.shape[0]
    onehot = np.asarray(pert_onehot, int)
    s = onehot.sum(1)
    arg = onehot.argmax(1)
    pert = np.array(["control" if s[i] == 0 else ("none" if s[i] >= 2 else f"guide_{int(arg[i])}")
                     for i in range(n)])
    tumor = np.zeros(n, bool)
    for fi in tumor_full_idx_1based:
        if 1 <= int(fi) <= n:
            tumor[int(fi) - 1] = True
    cell_type = np.where(tumor, "tumor", "other")
    return StandardData(
        X=X, coords=np.asarray(coords, float), perturbation=pert, cell_type=cell_type,
        batch=np.full(n, "tumors"), gene_names=list(gene_names), meta={"name": "Binan_tumors"},
    )


class BinanTumorsAdapter(DatasetAdapter):
    """Binan Perturb-FISH tumors (A375 melanoma + PBMC xenograft) finaltables -> StandardData,
    ALL cells (the immune-niche dataset). Follows the roadmap construction:
      X            = merfishcounttable.csv cols 4..553 (550 genes; cols 1-3 are id/total/volume)
      gene_names   = merfishcounttable_gene_mapping.csv (col_1based -> name)
      coords       = coordinates.csv (row-aligned, x,y)
      perturbation = allcellsPerturbationTable.csv (n_cells x 77 one-hot): no guide -> 'control',
                     single guide -> 'guide_<k>', multiplet -> 'none'
      cell_type    = 'tumor' for cells in tumorMerfish_index_xy.full_cell_index_1based, else 'other'

    FLAGGED refinements (model/明早问 Q2): the 77 guide columns are unnamed here (guide_<k>); a
    guide->gene-target key would name them. cell_type is coarse (tumor vs other) because only tumor
    cells carry a full-cell index; T cells are not separately indexed. Both are enhancements, not
    corrections — the adapter loads + scores correctly as a first cut."""

    def __init__(self, directory):
        self.directory = directory

    def load(self):
        import pandas as pd
        d = self.directory
        gm = pd.read_csv(d + "/merfishcounttable_gene_mapping.csv")
        gene_cols = (gm["merfishcounttable_col_1based"].astype(int) - 1).tolist()
        genes = gm["name"].astype(str).tolist()
        mc = pd.read_csv(d + "/merfishcounttable.csv", header=None)
        X = mc.iloc[:, gene_cols].to_numpy(float)
        coords = pd.read_csv(d + "/coordinates.csv", header=None).to_numpy(float)
        onehot = pd.read_csv(d + "/allcellsPerturbationTable.csv", header=None).to_numpy(int)
        tum = pd.read_csv(d + "/tumorMerfish_index_xy.csv")
        tumor_idx = tum["full_cell_index_1based"].astype(int).tolist()
        return _assemble_binan(X, coords, onehot, genes, tumor_idx)
