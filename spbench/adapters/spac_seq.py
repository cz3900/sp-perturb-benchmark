"""SPAC-seq adapter (thin): reads the processed cell-level .h5ad built once on the server by
``spac_seq_prep.build_sample`` / ``to_standard_data`` (see scripts on the server) into StandardData.

The heavy bin->cell geometry + guide calling lives in ``spac_seq_prep``; this adapter only reads the
result so the benchmark harness loads SPAC-seq like any other dataset. The processed .h5ad stores:
X = cell x gene expression; obs[perturbation, cell_type, batch]; obsm['spatial'] = cell centroids.
"""
import numpy as np
from .base import DatasetAdapter
from ..data import StandardData


class SpacSeqAdapter(DatasetAdapter):
    """`path`: processed cohort .h5ad (e.g. processed/subQ.h5ad or processed/lung.h5ad)."""

    def __init__(self, path, name="SPAC-seq"):
        self.path = path
        self.name = name

    def load(self) -> StandardData:
        import anndata as ad
        a = ad.read_h5ad(self.path)
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        return StandardData(
            X=X.astype(np.float32),
            coords=np.asarray(a.obsm["spatial"], dtype=float),
            perturbation=a.obs["perturbation"].to_numpy().astype(str),
            cell_type=a.obs["cell_type"].to_numpy().astype(str),
            batch=a.obs["batch"].to_numpy().astype(str),
            gene_names=list(a.var_names),
            meta={"name": a.uns.get("name", self.name)},
        )
