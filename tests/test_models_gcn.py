import numpy as np
from spbench.graph import build_knn_graph, neighbors_of
from spbench.models.gcn_prop import SimpleGCN

def test_prediction_independent_of_target_own_value(synth):
    edges = build_knn_graph(synth, k=8)
    m = SimpleGCN(hidden=16, epochs=3).fit(synth, edges)
    X2 = synth.X.copy()
    X2[5] += 100.0
    p1 = m.forward_numpy(synth.X, edges)[5]
    p2 = m.forward_numpy(X2, edges)[5]
    assert np.allclose(p1, p2, atol=1e-4)

def test_training_reduces_reconstruction_loss(synth):
    edges = build_knn_graph(synth, k=8)
    m = SimpleGCN(hidden=16, epochs=1)
    l0 = m._loss_once(synth, edges)
    m.fit(synth, edges)
    l1 = m._loss_once(synth, edges)
    assert l1 <= l0
