import numpy as np, h5py
from spbench.synthetic import make_synthetic
from spbench.graph import build_knn_graph
from spbench.models.celcomen_model import CelcomenModel

def _write(path, X, layer):
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=np.zeros_like(X))
        f.create_group("layers").create_dataset(layer, data=np.asarray(X, float))

def test_celcomen_predict_niche_and_external(tmp_path):
    from spbench.config import run_benchmark
    data = make_synthetic(0); edges = build_knn_graph(data, k=8)
    P = data.perturbations()[0]; p = tmp_path / f"{P}.h5ad"
    _write(str(p), data.X + 0.5, "counterfactual")
    model = CelcomenModel({P: str(p)})
    niche = model.predict_niche(data, P, edges)
    assert niche.ndim == 2 and niche.shape[1] == data.n_genes
    res = run_benchmark(data, perturbations=[P],
                        external_models={"Celcomen": model}, progress=False)
    assert "Celcomen" in res["compare"][P]["pcc"]
