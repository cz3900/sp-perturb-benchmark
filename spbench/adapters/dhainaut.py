import glob, os, csv
import numpy as np, h5py
from scipy.sparse import csc_matrix
from .base import DatasetAdapter
from ..data import StandardData

_NONE = {"periphery", "NA", "None", "nan", ""}


def _read_10x_h5(path):
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        ng, ns = (int(x) for x in m["shape"][:])
        X = csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=(ng, ns)).toarray().T
        dec = lambda a: [x.decode() if isinstance(x, bytes) else x for x in a]
        genes = dec(m["features/name"][:])
        barcodes = dec(m["barcodes"][:])
    return X.astype(float), genes, barcodes


def _read_positions(path):
    pos = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) >= 6:
                pos[row[0]] = (float(row[4]), float(row[5]))
    return pos


def _read_annotation(path):
    with open(path) as fh:
        return {r["barcode"]: r for r in csv.DictReader(fh)}


class DhainautAdapter(DatasetAdapter):
    """Perturb-map (Dhainaut 2022) GSE193460_RAW spaceranger -> StandardData, SPOT as the unit.
    perturbation: phenotypes base (split '_'): KP_* -> 'control', periphery/NA -> 'none', gene KO
    (Tgfbr2/Ifngr2/Jak2/...) -> that gene. cell_type = leiden_clusters (kmeans fallback). coords =
    full-res pixel (kNN graph is scale-invariant)."""

    def __init__(self, directory, max_files=None):
        self.directory = directory
        self.max_files = max_files

    def load(self):
        h5s = sorted(glob.glob(os.path.join(self.directory, "*_filtered_feature_bc_matrix.h5")))
        if self.max_files:
            h5s = h5s[: self.max_files]
        Xs, coords, pert, ct, batch, genes = [], [], [], [], [], None
        for h5 in h5s:
            stem = h5.replace("_filtered_feature_bc_matrix.h5", "")
            gsm = os.path.basename(stem)
            X, g, barcodes = _read_10x_h5(h5)
            pos = _read_positions(stem + "_tissue_positions_list.csv")
            ann = _read_annotation(stem + "_spot_annotation.csv")
            keep = [i for i, b in enumerate(barcodes) if b in pos]
            bcs = [barcodes[i] for i in keep]
            Xs.append(X[keep])
            coords.append(np.array([pos[b] for b in bcs], float))
            p, c = [], []
            for b in bcs:
                a = ann.get(b, {})
                pheno = (a.get("phenotypes") or "NA").strip()
                base = pheno.split("_", 1)[0]
                if base == "KP":
                    p.append("control")
                elif base in _NONE or pheno in _NONE:
                    p.append("none")
                else:
                    p.append(base)
                leiden = (a.get("leiden_clusters") or "NA").strip()
                c.append(leiden if leiden not in _NONE else (a.get("kmeans") or "NA").strip())
            pert.append(np.array(p)); ct.append(np.array(c))
            batch.append(np.full(len(bcs), gsm))
            if genes is None:
                genes = g
        return StandardData(
            X=np.vstack(Xs), coords=np.vstack(coords),
            perturbation=np.concatenate(pert).astype(str),
            cell_type=np.concatenate(ct).astype(str),
            batch=np.concatenate(batch).astype(str),
            gene_names=genes, meta={"name": "Dhainaut_2022"},
        )
