import numpy as np
from spbench.graph import build_knn_graph
from spbench.harness import fill_2x2, _control_reference_aggregate, _control_residuals
from spbench.compare import compare_to_baseline
from spbench.models.trivial_seed import TrivialSeed
from spbench.models.gaussian_prop import GaussianProp
from spbench.models.gcn_prop import SimpleGCN


def _niches(synth):
    edges = build_knn_graph(synth, k=8)
    X_ref = _control_reference_aggregate(synth, edges)
    resid = _control_residuals(synth)
    seed = TrivialSeed().fit(synth)
    base = GaussianProp().fit(synth, edges)
    learned = SimpleGCN(hidden=16, epochs=5).fit(synth, edges)
    g = fill_2x2(synth, "P0", edges, seed, base, learned, k_ref=5, X_ref=X_ref,
                 return_niches=True, residuals=resid)
    return g["_niches"], resid


def test_compare_returns_per_repeat_samples(synth):
    niches, resid = _niches(synth)
    out = compare_to_baseline(niches, residuals=resid, repeats=20)
    assert "e_samples" in out
    assert "model+learned" in out["e_samples"], "GCN (learned prop) must be a named method"
    for k, mean_e in out["e"].items():
        s = out["e_samples"][k]
        assert len(s) == 20
        assert np.isclose(np.nanmean(s), mean_e)
