import numpy as np
import pytest
from spbench.graph import neighbors_of

sq = pytest.importorskip("squidpy")  # 整文件依赖 squidpy,未装则跳过
from spbench.niche import build_spatial_graph


def test_returns_2xE_int_edges(synth):
    edges = build_spatial_graph(synth, n_neighs=6)
    assert edges.ndim == 2 and edges.shape[0] == 2
    assert edges.dtype.kind == "i"
    assert edges.shape[1] > 0


def test_edges_within_batch_only(synth):
    edges = build_spatial_graph(synth, n_neighs=6)
    b = synth.batch
    assert np.all(b[edges[0]] == b[edges[1]])


def test_no_self_loops(synth):
    edges = build_spatial_graph(synth, n_neighs=6)
    assert np.all(edges[0] != edges[1])


def test_neighbors_of_compatible(synth):
    """build_spatial_graph 的输出能直接喂 neighbors_of(同 graph.py 契约)。"""
    edges = build_spatial_graph(synth, n_neighs=6)
    nb = neighbors_of(0, edges)
    assert 0 not in nb and len(nb) > 0


def test_indices_in_range(synth):
    edges = build_spatial_graph(synth, n_neighs=6)
    assert edges.min() >= 0 and edges.max() < synth.n_cells
