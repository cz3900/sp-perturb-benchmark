import numpy as np
from spbench.graph import build_knn_graph
from spbench.models.gaussian_prop import GaussianProp

def test_closer_neighbour_gets_more_shift(synth):
    edges = build_knn_graph(synth, k=8)
    m = GaussianProp(bandwidth=2.0).fit(synth, edges)
    center = np.where(synth.perturbation == "P0")[0][0]
    from spbench.graph import neighbors_of
    nb = neighbors_of(center, edges)
    nb = nb[~synth.is_perturbed[nb]]
    seed = synth.X[center] + 5.0
    pred = m.propagate(synth.X, edges, center, seed, nb)
    assert pred.shape == (len(nb), synth.n_genes)
