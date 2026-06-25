import numpy as np
from .base import DatasetAdapter
from .cheng import _is_control_target
from ..data import StandardData


def _assemble_binan(X, coords, gene_names, tumor_pert_by_full_idx,
                    tumor_idx, immune_nb_idx, without_nb_idx):
    """Assemble the Binan tumors StandardData (option B: named guides; pure — file I/O in the adapter).

    The complete spatial substrate is the all-cells table (every cell has coords), so a perturbed
    tumor cell's niche graph includes its immune/stroma neighbours. Only TUMOR cells carry a guide
    (the screen perturbs tumor cells): `tumor_pert_by_full_idx` maps a tumor cell's 1-based full-cell
    index -> its NAMED guide gene (or 'control' for a no-guide tumor cell). Every other cell -> 'none'
    (bystander / non-tumor). cell_type = 'tumor' for `tumor_idx`, else 'other'. The immune-neighbour
    partition (a tumor-cell annotation, immune-nb ⊆ tumor) is carried in meta for stratified niche
    analysis (T cells are NOT a spatial unit here: different 114-gene panel, no coords)."""
    X = np.asarray(X, float)
    n = X.shape[0]
    pert = np.array(["none"] * n, dtype=object)
    for f, g in tumor_pert_by_full_idx.items():
        if 1 <= int(f) <= n:
            gv = str(g)
            # one of the 36 guides is literally named "Control" (non-targeting) -> 'control'
            pert[int(f) - 1] = "control" if _is_control_target(gv) else gv
    tumor = np.zeros(n, bool)
    for f in tumor_idx:
        if 1 <= int(f) <= n:
            tumor[int(f) - 1] = True
    cell_type = np.where(tumor, "tumor", "other")
    meta = {"name": "Binan_tumors",
            "immune_neighbor_idx": sorted(int(f) for f in immune_nb_idx),
            "immune_distal_idx": sorted(int(f) for f in without_nb_idx)}
    return StandardData(
        X=X, coords=np.asarray(coords, float), perturbation=pert.astype(str),
        cell_type=cell_type, batch=np.full(n, "tumors"), gene_names=list(gene_names), meta=meta,
    )


class BinanTumorsAdapter(DatasetAdapter):
    """Binan Perturb-FISH tumors (A375 melanoma + PBMC xenograft) finaltables -> StandardData
    (option B: all-cells spatial substrate + NAMED tumor guides).
      X            = merfishcounttable.csv cols 4..553 (550 genes; cols 1-3 are id/total/volume)
      gene_names   = merfishcounttable_gene_mapping.csv (col_1based -> name)
      coords       = coordinates.csv (row-aligned to merfishcounttable; every cell)
      perturbation = NAMED guide per tumor cell from tumorpooledperturbations.csv (36 guides,
                     row-labelled by gene; column j = tumor subset cell j -> full_cell_index via
                     tumorMerfish_index_xy). no-guide tumor cell -> 'control'; multiplet -> 'none';
                     non-tumor cell -> 'none' (bystander).
      cell_type    = 'tumor' (tumorMerfish_index_xy.full_cell_index) else 'other'
      meta         = immune_neighbor_idx / immune_distal_idx (with[out]immuneneighbor full indices;
                     a tumor-cell sub-annotation, since immune-nb ⊆ tumor)

    Note: T cells are not a spatial unit here (114-gene panel, no coords); the immune microenvironment
    enters via the all-cells spatial graph (tumor cells' neighbours) + the immune-neighbour meta."""

    def __init__(self, directory):
        self.directory = directory

    def load(self):
        import pandas as pd
        d = self.directory
        gm = pd.read_csv(d + "/merfishcounttable_gene_mapping.csv")
        gene_cols = (gm["merfishcounttable_col_1based"].astype(int) - 1).tolist()
        genes = gm["name"].astype(str).tolist()
        X = pd.read_csv(d + "/merfishcounttable.csv", header=None).iloc[:, gene_cols].to_numpy(float)
        coords = pd.read_csv(d + "/coordinates.csv", header=None).to_numpy(float)
        # tumor subset rows (in subset_row order) -> 1-based full-cell index
        tix = pd.read_csv(d + "/tumorMerfish_index_xy.csv")
        tumor_full = tix["full_cell_index_1based"].astype(int).tolist()
        # named guides: rows = 36 guide genes, cols = Cell_1..Cell_9432 (= tumor subset rows)
        tp = pd.read_csv(d + "/tumorpooledperturbations.csv", index_col=0)
        guide_names = [str(g) for g in tp.index]
        M = tp.to_numpy()                                              # (36, n_tumor)
        tumor_pert = {}
        for j, full in enumerate(tumor_full):
            col = M[:, j]
            s = int(np.nansum(col))
            tumor_pert[full] = ("control" if s == 0 else
                                ("none" if s >= 2 else guide_names[int(np.nanargmax(col))]))
        imm = set(pd.read_csv(d + "/withimmuneneighborMerfish_index_xy.csv")["full_cell_index_1based"].astype(int))
        wo = set(pd.read_csv(d + "/withoutimmuneneighborMerfish_index_xy.csv")["full_cell_index_1based"].astype(int))
        return _assemble_binan(X, coords, genes, tumor_pert, set(tumor_full), imm, wo)
