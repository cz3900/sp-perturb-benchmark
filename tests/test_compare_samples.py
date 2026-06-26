import numpy as np
from spbench.graph import build_knn_graph
from spbench.harness import fill_2x2, _control_reference_aggregate
from spbench.compare import compare_to_baseline, evaluate_seed
from spbench.models.trivial_seed import TrivialSeed
from spbench.models.gaussian_prop import GaussianProp
from spbench.models.gcn_prop import SimpleGCN


def _niches(synth):
    edges = build_knn_graph(synth, k=8)
    X_ref = _control_reference_aggregate(synth, edges)
    seed = TrivialSeed().fit(synth)
    base = GaussianProp().fit(synth, edges)
    learned = SimpleGCN(hidden=16, epochs=5).fit(synth, edges)
    g = fill_2x2(synth, "P0", edges, seed, base, learned, k_ref=5, X_ref=X_ref,
                 return_niches=True)
    return g["_niches"]


def test_compare_returns_pcc_and_mag_per_method(synth):
    niches = _niches(synth)
    out = compare_to_baseline(niches)
    assert set(out) == {"pcc", "mag", "n"}
    assert "model+learned" in out["pcc"], "GCN (learned prop) must be a named method"
    assert set(out["mag"]) == set(out["pcc"])
    for k, v in out["pcc"].items():
        if k == "null":
            assert np.isnan(v)            # flat no-effect shift -> no direction
        else:
            assert np.isfinite(v)


def test_evaluate_seed_returns_pcc_mse_mag(synth):
    niches = _niches(synth)
    out = evaluate_seed(niches)
    assert set(out) == {"pcc_delta", "mse", "mag", "n"}
    assert np.isfinite(out["pcc_delta"]) and out["mse"] >= 0.0
