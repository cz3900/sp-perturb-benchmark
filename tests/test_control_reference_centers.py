"""Direct, DISCRIMINATING unit tests for reference_aggregate.control_reference_centers.

This function carries the whole aggregate-control redesign contract: each perturbed center's
reference is ALL same-cell-type CONTROL cells (sample-level), with NO expression nearest-neighbour
matching (the retired match_reference_centers leakage source), falling back to all controls when a
type has none. The earlier suite only ever recomputed the expected value with the production
function itself (tautological — it would pass even if the leaky matcher were swapped back in), so
these tests pin the actual properties independently.
"""
import numpy as np
from spbench.reference import match_reference_centers
from spbench.reference_aggregate import control_reference_centers


def test_returns_same_type_controls_only(synth):
    """Every reference cell is a CONTROL of the SAME cell type as its center — no non-control,
    no wrong-type contamination."""
    centers = np.where(synth.perturbation == "P0")[0]
    refs = control_reference_centers(synth, centers)
    assert len(refs) == len(centers)
    for c, r in zip(centers, refs):
        assert len(r) > 0
        assert np.all(synth.is_control[r]), "reference contains a non-control cell"
        assert np.all(synth.cell_type[r] == synth.cell_type[c]), "reference mixes cell types"


def test_returns_all_same_type_controls_not_k(synth):
    """Aggregate control uses the WHOLE same-type control pool (computed independently), not a
    k-nearest subset."""
    centers = np.where(synth.perturbation == "P0")[0]
    refs = control_reference_centers(synth, centers)
    for c, r in zip(centers, refs):
        expected = np.where(synth.is_control & (synth.cell_type == synth.cell_type[c]))[0]
        assert set(r.tolist()) == set(expected.tolist())


def test_not_expression_nearest_neighbour(synth):
    """The discriminating property vs the retired match_reference_centers: for a cell type with
    more than k controls the aggregate pool is a strict SUPERSET of the k feature-nearest
    controls. Swapping the leaky expression-NN matcher back in would shrink the set and fail here."""
    centers = np.where(synth.perturbation == "P0")[0]
    agg = control_reference_centers(synth, centers)
    matched = match_reference_centers(synth, centers, k=5)
    saw_superset = False
    for c, a, m in zip(centers, agg, matched):
        n_type_ctrl = int((synth.is_control & (synth.cell_type == synth.cell_type[c])).sum())
        if n_type_ctrl > 5:
            assert len(a) == n_type_ctrl and len(a) > len(m)
            assert set(m.tolist()).issubset(set(a.tolist()))
            saw_superset = True
    assert saw_superset, "fixture has no cell type with >5 controls — cannot discriminate from k-NN"


def test_no_control_of_type_falls_back_to_all_controls(synth):
    """A perturbed center whose cell type has zero control cells falls back to ALL controls."""
    centers = np.where(synth.perturbation == "P0")[0]
    c = int(centers[0])
    cell_type = synth.cell_type.astype(object)
    cell_type[c] = "ghost_type"          # a type no control cell has
    synth.cell_type = cell_type
    refs = control_reference_centers(synth, np.array([c]))
    all_ctrl = np.where(synth.is_control)[0]
    assert set(refs[0].tolist()) == set(all_ctrl.tolist())
