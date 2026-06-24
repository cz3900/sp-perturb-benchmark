# Plan 5 — DhainautAdapter (Perturb-map Visium spot) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** `DhainautAdapter(directory)` reads the GSE193460_RAW spaceranger output (Perturb-map, Dhainaut 2022) into one `StandardData`, **spot as the unit**. Verified server structure (node03): per slice `GSM*_KP_N_{filtered_feature_bc_matrix.h5, tissue_positions_list.csv, scalefactors_json.json, spot_annotation.csv}`. TDD with a synthetic mini-spaceranger fixture (no server data needed); real-data smoke test is controller-run on the server after.

**Verified data structure:**
- `*_filtered_feature_bc_matrix.h5`: 10X format — `/matrix/{data,indices,indptr}` CSC, `/matrix/shape`=`[n_genes, n_spots]` (e.g. `[32289, 1906]`), `/matrix/features/name` (gene symbols), `/matrix/barcodes`. Matrix is genes×spots → transpose to spots×genes.
- `*_tissue_positions_list.csv`: **no header**, cols = `barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres`. Contains in- AND off-tissue spots; align to the filtered h5 barcodes.
- `*_spot_annotation.csv`: header `barcode, nCount_Spatial, nFeature_Spatial, kmeans, leiden_clusters, phenotypes`. `phenotypes` values e.g. `Tgfbr2_1`, `Jak2_1`, `KP_1-1`, `KP_1-2`, `KP_1-3`, `periphery`, `NA`.
- **Mapping**: `base = phenotypes.split('_', 1)[0]`. `KP` → `'control'`; `periphery`/`NA`/empty → `'none'`; otherwise (gene symbol e.g. `Tgfbr2`,`Ifngr2`,`Jak2`,`Socs1`) → that gene (KO). `cell_type = leiden_clusters` (fallback to `kmeans` when leiden is `NA`). `coords = (pxl_row_in_fullres, pxl_col_in_fullres)` (pixel coords; fine for the kNN graph — scale-invariant). `batch = GSM stem`.

**Tech Stack:** Python 3.11, numpy, h5py, scipy.sparse, pytest.

---

## File Structure
- `spbench/adapters/dhainaut.py` (new): `DhainautAdapter`.
- `spbench/adapters/__init__.py` (modify): register `"dhainaut"`.
- `tests/test_adapter_dhainaut.py` (new): synthetic-spaceranger fixture test.

---

### Task 1: DhainautAdapter + synthetic-fixture test

- [ ] **Step 1 — failing test** (`tests/test_adapter_dhainaut.py`). Build a tiny spaceranger fixture in `tmp_path` and assert the StandardData:
```python
import json, csv
import numpy as np, h5py
from scipy.sparse import csr_matrix
from spbench.adapters.dhainaut import DhainautAdapter

def _write_slice(d, stem, barcodes, X_spots_genes, genes, phenos, leidens, kmeans, extra_offtissue=1):
    # 10X filtered h5: matrix is genes x spots, CSC
    Xgs = np.asarray(X_spots_genes, float).T                      # genes x spots
    M = csr_matrix(Xgs).tocsc()
    p = d / f"{stem}_filtered_feature_bc_matrix.h5"
    with h5py.File(p, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("data", data=M.data.astype("float32"))
        m.create_dataset("indices", data=M.indices.astype("int64"))
        m.create_dataset("indptr", data=M.indptr.astype("int64"))
        m.create_dataset("shape", data=np.array(Xgs.shape, "int64"))
        m.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        feat = m.create_group("features")
        feat.create_dataset("name", data=np.array(genes, dtype="S"))
        feat.create_dataset("id", data=np.array(genes, dtype="S"))
    # tissue_positions_list.csv (no header); include extra off-tissue barcodes not in the h5
    with open(d / f"{stem}_tissue_positions_list.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        for i, b in enumerate(barcodes):
            w.writerow([b, 1, i, i, 100 + 10*i, 200 + 5*i])
        for j in range(extra_offtissue):
            w.writerow([f"OFF{j}-1", 0, 99, 99, 0, 0])             # off-tissue, must be dropped
    # spot_annotation.csv
    with open(d / f"{stem}_spot_annotation.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["barcode","nCount_Spatial","nFeature_Spatial","kmeans","leiden_clusters","phenotypes"])
        for b, ph, le, km in zip(barcodes, phenos, leidens, kmeans):
            w.writerow([b, 1000, 500, km, le, ph])
    json.dump({"spot_diameter_fullres": 32.0}, open(d / f"{stem}_scalefactors_json.json", "w"))

def test_dhainaut_adapter_maps_fields(tmp_path):
    genes = ["Tgfbr2", "Jak2", "Ptprc", "Cd8a"]
    bcs = ["A-1","B-1","C-1","D-1","E-1","F-1"]
    X = np.arange(6*4).reshape(6, 4).astype(float)
    phenos = ["Tgfbr2_1","KP_1-1","KP_1-2","Jak2_1","periphery","NA"]
    leidens = ["3","2","2","4","NA","NA"]
    kmeans  = ["tumor","tumor","tumor","tumor","normal","normal"]
    _write_slice(tmp_path, "GSM999_KP_1", bcs, X, genes, phenos, leidens, kmeans)
    data = DhainautAdapter(str(tmp_path)).load()
    assert data.X.shape == (6, 4)
    assert list(data.gene_names) == genes
    # perturbation mapping
    assert list(data.perturbation) == ["Tgfbr2","control","control","Jak2","none","none"]
    # cell_type = leiden (kmeans fallback when NA)
    assert list(data.cell_type) == ["3","2","2","4","normal","normal"]
    # off-tissue barcode dropped (6 spots, not 7)
    assert data.n_cells == 6
    assert data.coords.shape == (6, 2)
    # X aligned to barcode order (row 0 = first barcode's expression)
    assert np.allclose(data.X[0], X[0])
    # control + perturbed sets
    assert data.is_control.sum() == 2 and set(data.perturbations()) == {"Tgfbr2","Jak2"}
```

- [ ] **Step 2 — confirm FAIL** (`DhainautAdapter` undefined).

- [ ] **Step 3 — implement `spbench/adapters/dhainaut.py`:**
```python
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
```

- [ ] **Step 4 — register.** In `spbench/adapters/__init__.py`: `from .dhainaut import DhainautAdapter` and add `"dhainaut": DhainautAdapter` to `_REGISTRY`.

- [ ] **Step 5 — PASS + full suite + commit.** `python -m pytest tests/test_adapter_dhainaut.py -q` PASS; `python -m pytest -q` green. `git add -A && git commit -m "feat(adapter): DhainautAdapter (Perturb-map GSE193460_RAW spaceranger -> StandardData, spot unit) (Plan 5)"`

---

## Self-Review
- **Coverage**: roadmap Plan 5 mapping table → adapter fields + the fixture test asserts each (X/coords/perturbation/cell_type/gene_names/control/off-tissue-drop). Registry updated.
- **Server verification (controller, after merge)**: `DhainautAdapter('/home/yiru/.../GSE193460_RAW', max_files=1).load()` on node03 → run_benchmark(KO=Tgfbr2, control=KP) + plot_seed_prop → one figure. (Real data, can't run locally.)
- **Out of scope**: µm scaling (pixel coords suffice for kNN), deconvolution/D3 (optional, doc §4.4), the yiru 2000-panel version (rejected).
