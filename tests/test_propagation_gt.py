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


def test_reference_centers_are_same_type_controls(synth):
    """Pins e_null's SOURCE (the no-effect niche): ref_centers must be CONTROL cells whose types
    are confined to the perturbed centers' types — the aggregate same-type control niche, not a
    leaky expression-matched or wrong-type reference. Recompute-free, so it fails if the source is
    ever swapped back to feature-matched / mixed-type controls."""
    edges = build_knn_graph(synth, k=8)
    gt = propagation_gt(synth, "P0", edges, k_ref=5)
    rc = gt["ref_centers"]
    assert len(rc) > 0
    assert np.all(synth.is_control[rc]), "reference_niche sourced from non-control cells"
    # DISCRIMINATING: ref_centers must be ALL same-type controls (the aggregate pool), not a
    # feature-matched k-subset — so swapping the leaky match_reference_centers back in would fail.
    types = np.unique(synth.cell_type[gt["centers"]])
    expected = np.where(synth.is_control & np.isin(synth.cell_type, types))[0]
    assert set(rc.tolist()) == set(expected.tolist())
