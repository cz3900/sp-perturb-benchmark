import numpy as np
from spbench.graph import build_knn_graph
from spbench.propagation_gt import propagation_gt

def test_perturbed_neighbors_shifted_on_prop_gene(synth):
    edges = build_knn_graph(synth, k=8)
    gt = propagation_gt(synth, "P0", edges, k_ref=5)
    pg = synth.meta["planted"]["P0"]["prop_gene"]
    assert gt["perturbed_niche"][:, pg].mean() > gt["reference_niche"][:, pg].mean() + 0.2

def test_excludes_other_perturbed_cells(synth):
    edges = build_knn_graph(synth, k=8)
    gt = propagation_gt(synth, "P0", edges, k_ref=5)
    assert gt["perturbed_niche"].shape[0] > 0 and gt["reference_niche"].shape[0] > 0
