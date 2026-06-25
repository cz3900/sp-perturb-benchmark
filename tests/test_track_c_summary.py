import numpy as np, h5py
from spbench.synthetic import make_synthetic
from spbench.graph import build_knn_graph
from spbench.harness import fill_2x2
from spbench.compare import evaluate_seed
from spbench.models.seed_dump import SeedDumpModel
from spbench.models.gaussian_prop import GaussianProp

def _mock_dump(path, data, P):
    centers = np.where(data.perturbation == P)[0]
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=data.X[centers] + 0.2)        # finite shift
        f.create_group("obs").create_dataset("center_idx", data=centers.astype(np.int64))
    return centers

def test_three_seed_models_score_under_pcc_delta(tmp_path):
    data = make_synthetic(0); edges = build_knn_graph(data, k=8)
    P = data.perturbations()[0]
    base = GaussianProp().fit(data, edges)
    for name in ("cpa", "gears", "biolord"):
        dump = tmp_path / f"{name}_{P}_seed.h5ad"
        _mock_dump(str(dump), data, P)
        model = SeedDumpModel(name, {P: str(dump)}).fit(None)
        grid = fill_2x2(data, P, edges, model, base, base, return_niches=True)
        res = evaluate_seed(grid["_niches"])
        assert res["n"] > 0
        assert np.isfinite(res["mse"])
        # pcc_delta is finite or NaN-flat, never an exception
        assert np.isfinite(res["pcc_delta"]) or np.isnan(res["pcc_delta"])
