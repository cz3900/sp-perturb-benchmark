import numpy as np
from spbench.synthetic import make_synthetic
from spbench.graph import build_knn_graph
from spbench.external import score_external_niche
from spbench.models.mock_end_to_end import MockEndToEnd


def test_mock_end_to_end_beats_null():
    data = make_synthetic(0)
    edges = build_knn_graph(data, k=10)
    p = data.perturbations()[0]
    model = MockEndToEnd(noise=0.3, seed=0).fit(data, edges)
    niche_pred = model.predict_niche(data, p, edges)
    assert niche_pred.shape[1] == data.n_genes and len(niche_pred) > 0
    r = score_external_niche(data, p, edges, niche_pred, name="mock")
    assert "mock" in r["e"] and "mock" in r["e_samples"]
    assert np.isfinite(r["e"]["mock"])
    assert r["e"]["mock"] < r["e"]["null"]          # a near-observed model beats 'no effect'


def test_run_benchmark_threads_external_models():
    from spbench.config import run_benchmark
    from spbench.models.mock_end_to_end import MockEndToEnd
    data = make_synthetic(0)
    p = data.perturbations()[0]
    res = run_benchmark(data, perturbations=[p], k=10, k_ref=5,
                        gcn_kwargs={"hidden": 8, "epochs": 3}, progress=False,
                        external_models={"mock": MockEndToEnd(noise=0.3, seed=0)})
    c = res["compare"][p]
    assert "mock" in c["e"] and "mock" in c["e_samples"]
    assert c["e"]["mock"] < c["e"]["null"]


def test_plot_includes_external_box():
    import matplotlib; matplotlib.use("Agg")
    from spbench.config import run_benchmark
    from spbench.models.mock_end_to_end import MockEndToEnd
    from spbench.plotting import collect_niche_tier
    data = make_synthetic(0)
    p = data.perturbations()[0]
    res = run_benchmark(data, perturbations=[p], k=10, k_ref=5,
                        gcn_kwargs={"hidden": 8, "epochs": 3}, progress=False,
                        external_models={"mock": MockEndToEnd(noise=0.3, seed=0)})
    boxes, dashed = collect_niche_tier(res, "learned")
    assert any("mock" in k for k in boxes)        # external model shows as its own box on the end-to-end board
