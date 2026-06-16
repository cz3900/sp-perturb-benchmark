import glob, os
import numpy as np, h5py
from .base import DatasetAdapter
from ..data import StandardData, UNLABELED

def _read_cat(grp):
    cats = [c.decode() if isinstance(c, bytes) else c for c in grp["categories"][:]]
    codes = grp["codes"][:]
    cats = np.array(cats + ["<NA>"], dtype=object)
    return cats[np.where(codes < 0, len(cats) - 1, codes)]

class SaundersAdapter(DatasetAdapter):
    """Reads Saunders .h5mu files. Pools all slices in `directory`.
    Maps singlet_gene '' -> 'none', keeps 'control' as control."""
    def __init__(self, directory: str, max_files: int | None = None):
        self.directory = directory
        self.max_files = max_files

    def load(self) -> StandardData:
        files = sorted(glob.glob(os.path.join(self.directory, "*.h5mu")))
        if self.max_files:
            files = files[: self.max_files]
        Xs, coords, pert, ct, batch, genes = [], [], [], [], [], None
        for f in files:
            with h5py.File(f, "r") as h:
                rna = h["mod/rna"]
                Xs.append(rna["layers/raw_scaled"][:])
                coords.append(rna["obsm/spatial"][:])
                sg = _read_cat(rna["obs/singlet_gene"])
                pert.append(np.where(sg == "", UNLABELED, sg))
                ct.append(_read_cat(rna["obs/cell_type"]))
                batch.append(np.full(Xs[-1].shape[0], os.path.basename(f)))
                if genes is None:
                    genes = [g.decode() if isinstance(g, bytes) else g
                             for g in rna["var/_index"][:]]
        return StandardData(
            X=np.vstack(Xs), coords=np.vstack(coords),
            perturbation=np.concatenate(pert).astype(str),
            cell_type=np.concatenate(ct).astype(str),
            batch=np.concatenate(batch).astype(str),
            gene_names=genes, meta={"name": "Saunders_2025"},
        )
