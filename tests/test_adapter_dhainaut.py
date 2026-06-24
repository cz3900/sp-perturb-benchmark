import json, csv
import numpy as np, h5py
from scipy.sparse import csr_matrix
from spbench.adapters.dhainaut import DhainautAdapter

def _write_slice(d, stem, barcodes, X_spots_genes, genes, phenos, leidens, kmeans, extra_offtissue=1):
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
    with open(d / f"{stem}_tissue_positions_list.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        for i, b in enumerate(barcodes):
            w.writerow([b, 1, i, i, 100 + 10*i, 200 + 5*i])
        for j in range(extra_offtissue):
            w.writerow([f"OFF{j}-1", 0, 99, 99, 0, 0])
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
    assert list(data.perturbation) == ["Tgfbr2","control","control","Jak2","none","none"]
    assert list(data.cell_type) == ["3","2","2","4","normal","normal"]
    assert data.n_cells == 6
    assert data.coords.shape == (6, 2)
    assert np.allclose(data.X[0], X[0])
    assert data.is_control.sum() == 2 and set(data.perturbations()) == {"Tgfbr2","Jak2"}
