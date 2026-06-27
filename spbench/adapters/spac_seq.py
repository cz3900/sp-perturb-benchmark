"""SPAC-seq adapter (thin): reads the processed cell-level .h5mu built once on the server by
`spac_seq_prep.assemble_mudata` (see build_cells.py on the server) into StandardData.

The .h5mu has two modalities: mod/rna (cell x gene expression -- the modelling X) and mod/guide
(cell x 1520 guide UMI counts, kept for re-thresholding). This adapter reads mod/rna; obs carries
perturbation / cell_type / batch and obsm['spatial'] = cell centroids. Like the saunders/shen
adapters, SPAC-seq then loads like any other dataset; the heavy bin->cell geometry lives in
spac_seq_prep.
"""
import numpy as np
from .base import DatasetAdapter
from ..data import StandardData


class SpacSeqAdapter(DatasetAdapter):
    """`path`: processed cohort .h5mu (e.g. processed/subQ.h5mu or processed/lung.h5mu)."""

    def __init__(self, path, name="SPAC-seq"):
        self.path = path
        self.name = name

    def load(self) -> StandardData:
        import mudata
        md = mudata.read_h5mu(self.path)
        rna = md.mod["rna"]
        X = rna.X.toarray() if hasattr(rna.X, "toarray") else np.asarray(rna.X)
        name = md.uns.get("name", self.name) if getattr(md, "uns", None) is not None else self.name
        return StandardData(
            X=X.astype(np.float32),
            coords=np.asarray(rna.obsm["spatial"], dtype=float),
            perturbation=rna.obs["perturbation"].to_numpy().astype(str),
            cell_type=rna.obs["cell_type"].to_numpy().astype(str),
            batch=rna.obs["batch"].to_numpy().astype(str),
            gene_names=list(rna.var_names),
            meta={"name": name},
        )
