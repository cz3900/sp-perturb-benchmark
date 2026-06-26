import numpy as np
from spbench.harness import _control_reference, _control_reference_aggregate
from spbench.reference_aggregate import aggregate_control
from spbench.graph import build_knn_graph


def test_aggregate_ref_shape(synth):
    edges = build_knn_graph(synth, k=10)
    X_ref = _control_reference_aggregate(synth, edges)
    assert X_ref.shape == (synth.n_cells, synth.n_genes)
    assert np.all(np.isfinite(X_ref))


def test_each_cell_maps_to_its_celltype_aggregate(synth):
    edges = build_knn_graph(synth, k=10)
    agg = aggregate_control(synth, edges)
    X_ref = _control_reference_aggregate(synth, edges)
    for ct in np.unique(synth.cell_type):
        rows = synth.cell_type == ct
        # every cell of this type shares the same aggregate expr vector
        expected = np.broadcast_to(agg.expr[ct], X_ref[rows].shape)
        np.testing.assert_allclose(X_ref[rows], expected, atol=1e-6)


def test_matches_legacy_control_reference_values(synth):
    # the per-cell-type control-mean content must equal the legacy reference
    # (aggregate version only differs in HOW it is sourced, not the expr numbers)
    edges = build_knn_graph(synth, k=10)
    legacy = _control_reference(synth)
    agg = _control_reference_aggregate(synth, edges)
    np.testing.assert_allclose(agg, legacy, atol=1e-6)


def test_fill_2x2_accepts_aggregate_xref(synth):
    # smoke: passing the aggregate X_ref through fill_2x2 yields a bounded PCC-delta grid.
    # registry get_model returns a CLASS -> instantiate AND .fit() before use.
    from spbench.harness import fill_2x2
    from spbench.models import get_model
    edges = build_knn_graph(synth, k=10)
    X_ref = _control_reference_aggregate(synth, edges)
    seed = get_model("trivial_seed")().fit(synth)
    base = get_model("gaussian_prop")().fit(synth, edges)
    learned = get_model("gaussian_prop")().fit(synth, edges)
    grid = fill_2x2(synth, "P0", edges, seed, base, learned, X_ref=X_ref)
    for cell in ("1", "2", "3", "4"):
        v = grid[cell]["pcc_prop"]
        assert np.isnan(v) or -1.0 <= v <= 1.0
