import numpy as np
import pytest
from spbench.graph import build_knn_graph, neighbors_of
from spbench.niche import compute_niche_composition, n_hop_neighbors


def test_shape_is_n_by_C(synth):
    edges = build_knn_graph(synth, k=6)
    comp = compute_niche_composition(synth, edges)
    C = len(np.unique(synth.cell_type))
    assert comp.shape == (synth.n_cells, C)


def test_rows_are_simplex(synth):
    """每个有邻居的细胞:组成向量非负且和为 1。"""
    edges = build_knn_graph(synth, k=6)
    comp = compute_niche_composition(synth, edges)
    has_nb = comp.sum(1) > 0
    assert np.all(comp >= 0)
    assert np.allclose(comp[has_nb].sum(1), 1.0)


def test_isolated_cell_is_zero_row():
    """无邻居的细胞 -> 全零行(不是 NaN)。"""
    from spbench.synthetic import make_synthetic
    data = make_synthetic(seed=0)
    edges = np.zeros((2, 0), dtype=int)   # 空图:人人无邻居
    comp = compute_niche_composition(data, edges)
    assert comp.shape[0] == data.n_cells
    assert np.all(comp == 0)


def test_composition_matches_manual_count(synth):
    """对某中心,手算邻居的 cell_type 频率应与组成向量逐项相等。"""
    edges = build_knn_graph(synth, k=6)
    cats = sorted(np.unique(synth.cell_type).tolist())
    comp = compute_niche_composition(synth, edges)
    c = 0
    nb = neighbors_of(c, edges)
    expect = np.array([(synth.cell_type[nb] == ct).sum() for ct in cats], float)
    expect = expect / expect.sum()
    assert np.allclose(comp[c], expect)


def test_n_hop_expands_neighborhood(synth):
    """2-hop 邻域应是 1-hop 的超集(且严格更大,网格图上成立)。"""
    edges = build_knn_graph(synth, k=6)
    nb1 = set(n_hop_neighbors(0, edges, hops=1).tolist())
    nb2 = set(n_hop_neighbors(0, edges, hops=2).tolist())
    assert nb1.issubset(nb2)
    assert len(nb2) > len(nb1)
    assert 0 not in nb2          # 中心自身不计入


def test_hops_changes_composition(synth):
    """尺度可配:hops=2 给出与 hops=1 不同的组成(尺度敏感性 sweep 的基础)。"""
    edges = build_knn_graph(synth, k=6)
    c1 = compute_niche_composition(synth, edges, hops=1)
    c2 = compute_niche_composition(synth, edges, hops=2)
    assert c1.shape == c2.shape
    assert not np.allclose(c1, c2)
