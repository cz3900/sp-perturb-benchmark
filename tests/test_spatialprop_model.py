import numpy as np, h5py
from spbench.synthetic import make_synthetic
from spbench.graph import build_knn_graph
from spbench.models.spatialprop_model import SpatialPropModel, read_h5ad_layer


def _write_tempered(path, X):
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=np.zeros_like(X))                  # raw slot unused by loader
        f.create_group("layers").create_dataset("predicted_tempered", data=np.asarray(X, float))


def test_read_h5ad_layer(tmp_path):
    p = tmp_path / "p.h5ad"; X = np.arange(12.).reshape(4, 3)
    _write_tempered(str(p), X)
    assert np.allclose(read_h5ad_layer(str(p), "predicted_tempered"), X)


def test_predict_niche_selects_bystander_neighbours(tmp_path):
    data = make_synthetic(0); edges = build_knn_graph(data, k=8)
    P = data.perturbations()[0]
    p = tmp_path / f"{P}.h5ad"
    _write_tempered(str(p), data.X + 0.5)                             # mock tempered prediction
    model = SpatialPropModel({P: str(p)})
    niche = model.predict_niche(data, P, edges)
    # bystander niche: (m, G), m >= 0, only non-perturbed neighbours of perturbed centers
    assert niche.ndim == 2 and niche.shape[1] == data.n_genes


def test_spatialprop_scores_as_external(tmp_path):
    from spbench.config import run_benchmark
    from spbench.plotting import summary_table
    data = make_synthetic(0)
    P = data.perturbations()[0]
    p = tmp_path / f"{P}.h5ad"; _write_tempered(str(p), data.X + 0.5)
    model = SpatialPropModel({P: str(p)})
    res = run_benchmark(data, perturbations=[P], external_models={"SpatialProp": model}, progress=False)
    assert "SpatialProp" in res["compare"][P]["pcc"]
    assert "niche_SpatialProp" in summary_table(res)[0]
