# /Users/cz/Documents/ZengLab/model/sp-perturb-benchmark/tests/test_reference_aggregate.py
import numpy as np
from spbench.reference_aggregate import aggregate_control
from spbench.graph import build_knn_graph


def test_returns_per_celltype_keys(synth):
    edges = build_knn_graph(synth, k=10)
    agg = aggregate_control(synth, edges)
    cts = set(np.unique(synth.cell_type).tolist())
    # every cell type with >=1 control cell must appear in all three dicts
    for ct in cts:
        has_ctrl = bool((synth.is_control & (synth.cell_type == ct)).any())
        if has_ctrl:
            assert ct in agg.expr
            assert ct in agg.niche_comp
            assert ct in agg.niche_expr


def test_expr_is_control_celltype_mean(synth):
    edges = build_knn_graph(synth, k=10)
    agg = aggregate_control(synth, edges)
    for ct in np.unique(synth.cell_type):
        m = synth.is_control & (synth.cell_type == ct)
        if not m.any():
            continue
        expected = synth.X[m].mean(0)
        np.testing.assert_allclose(agg.expr[ct], expected, rtol=1e-6, atol=1e-6)
    # shape == n_genes
    any_ct = next(iter(agg.expr))
    assert agg.expr[any_ct].shape == (synth.n_genes,)


def test_niche_composition_is_simplex(synth):
    edges = build_knn_graph(synth, k=10)
    agg = aggregate_control(synth, edges)
    cell_types = agg.cell_types
    # composition axis order is the sorted unique cell types
    assert list(cell_types) == sorted(cell_types)
    for ct, comp in agg.niche_comp.items():
        assert comp.shape == (len(cell_types),)
        assert np.all(comp >= -1e-9)
        np.testing.assert_allclose(comp.sum(), 1.0, atol=1e-6)


def test_niche_expr_shape_and_finiteness(synth):
    edges = build_knn_graph(synth, k=10)
    agg = aggregate_control(synth, edges)
    for ct, ne in agg.niche_expr.items():
        assert ne.shape == (synth.n_genes,)
        assert np.all(np.isfinite(ne))


def test_global_fallback_for_celltype_without_control():
    # build data where one cell type ("B") has zero control cells -> global control mean fallback
    from spbench.data import StandardData, CONTROL, UNLABELED
    rng = np.random.default_rng(1)
    n = 60
    coords = rng.uniform(0, 10, size=(n, 2))
    X = rng.normal(0, 1, size=(n, 5))
    ct = np.array(["A"] * 40 + ["B"] * 20, dtype=object)
    pert = np.full(n, UNLABELED, dtype=object)
    pert[:15] = CONTROL                      # controls only among type A
    data = StandardData(
        X=X, coords=coords, perturbation=pert.astype(str),
        cell_type=ct.astype(str), batch=np.full(n, "s0"),
        gene_names=[f"g{i}" for i in range(5)], meta={},
    )
    edges = build_knn_graph(data, k=8)
    agg = aggregate_control(data, edges)
    # B has no control cells -> expr falls back to the GLOBAL control mean, present & finite
    assert "B" in agg.expr
    gmean = X[data.is_control].mean(0)
    np.testing.assert_allclose(agg.expr["B"], gmean, atol=1e-6)
    assert np.all(np.isfinite(agg.niche_expr["B"]))
    np.testing.assert_allclose(agg.niche_comp["B"].sum(), 1.0, atol=1e-6)
