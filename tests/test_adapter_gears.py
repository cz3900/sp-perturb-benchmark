import numpy as np, h5py
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.gears_export import export_to_gears_h5
from spbench.adapters.counts_export import build_counts_X

def _d():
    counts = np.array([[2, 4, 0], [0, 6, 2], [4, 0, 8], [2, 2, 2]], float)
    return StandardData(
        X=np.zeros((4, 3)), coords=np.zeros((4, 2)),
        perturbation=np.array(["GeneA", CONTROL, UNLABELED, "GeneA"]),
        cell_type=np.array(["T", "T", "B", "B"]), batch=np.array(["s1"] * 4),
        gene_names=["g1", "g2", "g3"], meta={"counts": counts})

def test_gears_condition_and_var_gene_name(tmp_path):
    d = _d(); X = build_counts_X(d)
    export_to_gears_h5(d, "GeneA", X, str(tmp_path / "GeneA.h5ad"))
    with h5py.File(tmp_path / "GeneA.h5ad", "r") as f:
        cond = list(np.array(f["obs"]["condition"]).astype(str))
        assert cond == ["GeneA+ctrl", "ctrl", "GeneA+ctrl"]      # GEARS condition convention
        assert "gene_name" in f["var"]                            # singular key GEARS expects
