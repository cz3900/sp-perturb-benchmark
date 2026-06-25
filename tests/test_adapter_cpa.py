import numpy as np, h5py
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.cpa_export import export_to_cpa_h5
from spbench.adapters.counts_export import build_counts_X
from spbench.models.seed_dump import SeedDumpModel
from spbench.harness import fill_2x2
from spbench.compare import evaluate_seed
from spbench.graph import build_knn_graph

def _counts_data():
    rng = np.random.default_rng(3)
    n, g = 40, 6
    counts = rng.integers(0, 20, size=(n, g)).astype(float)
    pert = np.array([UNLABELED] * n, dtype=object); pert[:6] = "GeneA"; pert[6:18] = CONTROL
    return StandardData(
        X=rng.normal(size=(n, g)), coords=rng.uniform(0, 100, size=(n, 2)),
        perturbation=pert.astype(str),
        cell_type=np.where(np.arange(n) % 2 == 0, "T", "B").astype(str),
        batch=np.array(["s1"] * n), gene_names=[f"g{i}" for i in range(g)],
        meta={"counts": counts})

def test_cpa_export_uses_stimulated_control_and_raw_counts(tmp_path):
    d = _counts_data(); X = build_counts_X(d)
    info = export_to_cpa_h5(d, "GeneA", X, str(tmp_path / "GeneA.h5ad"))
    assert info["n_stim"] == 6 and info["n_ctrl"] == 12
    with h5py.File(tmp_path / "GeneA.h5ad", "r") as f:
        cond = set(np.array(f["obs"]["condition"]).astype(str))
        assert cond == {"stimulated", "control"}
        assert "cell_type" in f["obs"] and "gene_names" in f["var"]
        assert np.allclose(np.array(f["X"]), np.round(np.array(f["X"])))   # raw integer counts

def test_cpa_seed_dump_scores_end_to_end(tmp_path):
    d = _counts_data(); edges = build_knn_graph(d, k=6); P = "GeneA"
    centers = np.where(d.perturbation == P)[0]
    # mock CPA offline output: per-center aligned seed (here a finite arbitrary shift)
    dump = tmp_path / f"{P}_seed.h5ad"
    with h5py.File(dump, "w") as f:
        f.create_dataset("X", data=d.X[centers] + 0.3)
        f.create_group("obs").create_dataset("center_idx", data=centers.astype(np.int64))
    model = SeedDumpModel("cpa", {P: str(dump)}).fit(None)

    class _IdProp:
        name = "id"
        def fit(self, t, e): return self
        def propagate(self, Xref, e, c, s, nb): return Xref[nb]
    grid = fill_2x2(d, P, edges, model, _IdProp(), _IdProp(), return_niches=True)
    res = evaluate_seed(grid["_niches"])
    assert res["n"] == len(centers)
    assert np.isfinite(res["mse"])
