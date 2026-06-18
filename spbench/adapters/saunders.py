import glob, os
import numpy as np, h5py
from .base import DatasetAdapter
from ..data import StandardData, UNLABELED

def _read_cat(grp):
    cats = [c.decode() if isinstance(c, bytes) else c for c in grp["categories"][:]]
    codes = grp["codes"][:]
    cats = np.array(cats + ["<NA>"], dtype=object)
    return cats[np.where(codes < 0, len(cats) - 1, codes)]

def _read_matrix(node):
    """Read an expression matrix h5 node — a dense dataset OR an AnnData CSR/CSC sparse group."""
    if isinstance(node, h5py.Dataset):
        return node[:]
    from scipy.sparse import csr_matrix, csc_matrix
    data, indices, indptr = node["data"][:], node["indices"][:], node["indptr"][:]
    shape = tuple(int(s) for s in node.attrs["shape"])
    enc = str(node.attrs.get("encoding-type", "csr_matrix"))
    M = (csc_matrix if "csc" in enc else csr_matrix)((data, indices, indptr), shape=shape)
    return M.toarray()

class SaundersAdapter(DatasetAdapter):
    """Reads Saunders .h5mu files. Pools all slices in `directory`.
    Maps singlet_gene '' -> 'none', keeps 'control' as control.

    `counts_layer`: if given, ALSO load a raw-count matrix into meta['counts'] (aligned cell order),
    for exporting to models that need integer counts (e.g. CONCERT, which rejects the scaled X).
    "X" reads mod/rna/X; otherwise reads mod/rna/layers/<counts_layer>. Confirm the layer name by
    inspecting the .h5mu — data.X here is layers/raw_scaled (z-scored), not counts."""
    def __init__(self, directory: str, max_files: int | None = None, counts_layer: str | None = None):
        self.directory = directory
        self.max_files = max_files
        self.counts_layer = counts_layer

    def load(self) -> StandardData:
        files = sorted(glob.glob(os.path.join(self.directory, "*.h5mu")))
        if self.max_files:
            files = files[: self.max_files]
        Xs, coords, pert, ct, batch, genes = [], [], [], [], [], None
        counts = [] if self.counts_layer else None
        for f in files:
            with h5py.File(f, "r") as h:
                rna = h["mod/rna"]
                Xs.append(rna["layers/raw_scaled"][:])
                if counts is not None:
                    node = rna["X"] if self.counts_layer == "X" else rna[f"layers/{self.counts_layer}"]
                    counts.append(_read_matrix(node))
                coords.append(rna["obsm/spatial"][:])
                sg = _read_cat(rna["obs/singlet_gene"])
                pert.append(np.where(sg == "", UNLABELED, sg))
                ct.append(_read_cat(rna["obs/cell_type"]))
                batch.append(np.full(Xs[-1].shape[0], os.path.basename(f)))
                if genes is None:
                    genes = [g.decode() if isinstance(g, bytes) else g
                             for g in rna["var/_index"][:]]
        meta = {"name": "Saunders_2025"}
        if counts is not None:
            meta["counts"] = np.vstack(counts)
        return StandardData(
            X=np.vstack(Xs), coords=np.vstack(coords),
            perturbation=np.concatenate(pert).astype(str),
            cell_type=np.concatenate(ct).astype(str),
            batch=np.concatenate(batch).astype(str),
            gene_names=genes, meta=meta,
        )
