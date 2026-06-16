import numpy as np
from spbench.graph import build_knn_graph, neighbors_of

def test_edges_within_batch_only(synth):
    edges = build_knn_graph(synth, k=6)
    b = synth.batch
    assert np.all(b[edges[0]] == b[edges[1]])

def test_neighbors_exclude_self(synth):
    edges = build_knn_graph(synth, k=6)
    nb = neighbors_of(0, edges)
    assert 0 not in nb and len(nb) > 0
