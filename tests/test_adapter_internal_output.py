# tests/test_adapter_internal_output.py
import numpy as np
import pytest
from spbench.prediction import StandardPrediction
from spbench.adapters.internal_output import to_prediction


def test_seed_only_fills_d1():
    seed_out = np.array([[1.0, 3.0], [3.0, 5.0]])      # (m=2 ref cells, G=2)
    p = to_prediction(seed_out=seed_out)
    assert isinstance(p, StandardPrediction)
    assert np.allclose(p.d1, [2.0, 4.0])
    assert p.d2 is None
    assert p.d3 is None
    assert p.covered_dims() == ("D1",)


def test_prop_only_fills_d2():
    prop_out = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])  # (3 neighbours, G=2)
    p = to_prediction(prop_out=prop_out)
    assert p.d1 is None
    assert np.allclose(p.d2, [2.0, 4.0])
    assert p.d3 is None
    assert p.covered_dims() == ("D2",)


def test_seed_and_prop_fills_d1_d2():
    seed_out = np.array([[1.0, 1.0], [3.0, 3.0]])
    prop_out = np.array([[0.0, 0.0], [4.0, 8.0]])
    p = to_prediction(seed_out=seed_out, prop_out=prop_out)
    assert np.allclose(p.d1, [2.0, 2.0])
    assert np.allclose(p.d2, [2.0, 4.0])
    assert p.d3 is None
    assert p.covered_dims() == ("D1", "D2")


def test_d3_composition_normalized_to_simplex():
    seed_out = np.array([[2.0, 2.0]])
    comp = np.array([1.0, 1.0, 2.0])
    p = to_prediction(seed_out=seed_out, composition=comp)
    assert np.allclose(p.d3, [0.25, 0.25, 0.5])
    assert pytest.approx(p.d3.sum()) == 1.0
    assert p.covered_dims() == ("D1", "D3")


def test_d3_already_simplex_passthrough():
    comp = np.array([0.2, 0.3, 0.5])
    p = to_prediction(composition=comp)
    assert np.allclose(p.d3, [0.2, 0.3, 0.5])
    assert p.covered_dims() == ("D3",)


def test_empty_prop_array_yields_none_d2():
    prop_out = np.zeros((0, 4))
    p = to_prediction(prop_out=prop_out)
    assert p.d2 is None
    assert p.covered_dims() == ()


def test_nothing_supplied_is_empty_prediction():
    p = to_prediction()
    assert p.covered_dims() == ()


def test_zero_sum_composition_raises():
    with pytest.raises(ValueError):
        to_prediction(composition=np.zeros(3))


def test_aggregates_1d_seed_passthrough():
    p = to_prediction(seed_out=np.array([5.0, 7.0]))
    assert np.allclose(p.d1, [5.0, 7.0])
    assert p.covered_dims() == ("D1",)


def test_end_to_end_with_real_models(synth):
    from spbench.graph import build_knn_graph, neighbors_of
    from spbench.models.trivial_seed import TrivialSeed
    from spbench.models.gaussian_prop import GaussianProp
    edges = build_knn_graph(synth)
    seed = TrivialSeed().fit(synth)
    prop = GaussianProp().fit(synth, edges)
    pert = synth.perturbations()[0]
    centers = np.where(synth.perturbation == pert)[0]
    c = int(centers[0])
    nb = neighbors_of(c, edges)
    assert len(nb) > 0
    seed_out = seed.predict_seed(pert, synth.X[centers])
    seed_state = seed_out.mean(0)
    prop_out = prop.propagate(synth.X, edges, c, seed_state, nb)
    p = to_prediction(seed_out=seed_out, prop_out=prop_out)
    assert p.d1.shape == (synth.n_genes,)
    assert p.d2.shape == (synth.n_genes,)
    assert p.covered_dims() == ("D1", "D2")
