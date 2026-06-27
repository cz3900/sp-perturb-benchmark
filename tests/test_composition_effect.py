# tests/test_composition_effect.py
import numpy as np
import pytest

from spbench.composition import (
    build_adjacency, neighbor_composition, composition_effect, niche_composition,
    niche_effect_board,
)
from spbench.data import StandardData
from spbench.synthetic import make_synthetic_with_effects
from spbench.graph import build_knn_graph


def _line_edges():
    # 4 cells in a line: 0-1-2-3 (undirected, as [src,dst] both ways)
    return np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])


def _toy_data(cell_type, pert=None):
    n = len(cell_type)
    pert = pert if pert is not None else np.array(["none"] * n)
    return StandardData(X=np.zeros((n, 1), np.float32), coords=np.zeros((n, 2)),
                        perturbation=np.asarray(pert, object).astype(str),
                        cell_type=np.asarray(cell_type, object),
                        batch=np.array(["s"] * n), gene_names=["g"])


# --- adjacency ------------------------------------------------------------

def test_build_adjacency_neighbors():
    adj = build_adjacency(_line_edges(), 4)
    assert sorted(adj[1].tolist()) == [0, 2]
    assert sorted(adj[0].tolist()) == [1]
    assert adj.get(99, np.empty(0, int)).size == 0     # missing node -> empty


# --- all-neighbor composition (decision #3) -------------------------------

def test_neighbor_composition_uses_all_neighbors():
    # center 1's neighbors are 0 and 2 (types A, A) -> composition all A,
    # INCLUDING neighbours regardless of their perturbation status.
    data = _toy_data(["A", "B", "A", "B"])
    adj = build_adjacency(_line_edges(), 4)
    cats = np.array(["A", "B"], object)
    assert neighbor_composition(data, np.array([1]), adj, cats) == pytest.approx([1.0, 0.0])
    # center 2's neighbours are 1,3 (B,B)
    assert neighbor_composition(data, np.array([2]), adj, cats) == pytest.approx([0.0, 1.0])


# --- per-guide effect: two baselines + permutation ------------------------

def test_composition_effect_has_both_baselines_and_keys():
    data = make_synthetic_with_effects(seed=0)
    adj = build_adjacency(build_knn_graph(data, k=8), data.n_cells)
    r = composition_effect(data, "SIG", adj, n_perm=200, seed=0)
    for k in ("guide", "n_ko", "overlap_ntc", "overlap_pooled", "p_value", "delta_ntc", "cats"):
        assert k in r
    assert 0.0 <= r["overlap_ntc"] <= 1.0 and 0.0 <= r["overlap_pooled"] <= 1.0
    assert r["n_ko"] > 0


def test_composition_effect_detects_d3_shift_sig_vs_inert():
    data = make_synthetic_with_effects(seed=0)
    adj = build_adjacency(build_knn_graph(data, k=8), data.n_cells)
    sig = composition_effect(data, "SIG", adj, n_perm=0)
    inert = composition_effect(data, "INERT", adj, n_perm=0)
    # SIG plants a niche cell-type enrichment -> its niche overlaps the NTC niche LESS
    assert sig["overlap_ntc"] < inert["overlap_ntc"]


def test_composition_effect_permutation_significance():
    data = make_synthetic_with_effects(seed=0)
    adj = build_adjacency(build_knn_graph(data, k=8), data.n_cells)
    sig = composition_effect(data, "SIG", adj, n_perm=300, seed=0)
    inert = composition_effect(data, "INERT", adj, n_perm=300, seed=0)
    assert sig["p_value"] < 0.05        # planted effect -> significant
    assert inert["p_value"] > 0.05      # no effect -> not significant


def test_composition_effect_pooled_baseline_differs_from_ntc():
    # pooled baseline (all perturbed niches) is a different reference than NTC,
    # so the two overlaps should generally not be identical for an effect guide.
    data = make_synthetic_with_effects(seed=0)
    adj = build_adjacency(build_knn_graph(data, k=8), data.n_cells)
    r = composition_effect(data, "SIG", adj, n_perm=0)
    assert r["overlap_ntc"] != pytest.approx(r["overlap_pooled"], abs=1e-6)


# --- board / runner -------------------------------------------------------

def test_niche_effect_board_runs_and_sorts():
    data = make_synthetic_with_effects(seed=0)
    edges = build_knn_graph(data, k=8)
    board = niche_effect_board(data, edges, min_ko=5, n_perm=100, seed=0)
    rows = board["rows"]
    assert board["n_guides"] == len(rows) >= 2
    assert "SIG" in [r["guide"] for r in rows]
    ov = [r["overlap_pooled"] for r in rows]
    assert ov == sorted(ov)                              # sorted by guide-specific shift (headline)
    assert all("p_value" in r for r in rows)


def test_niche_effect_board_guards_single_cell_type():
    # Cheng-like single cell line -> composition is degenerate, board returns empty + note.
    data = make_synthetic_with_effects(seed=0)
    data.cell_type[:] = "OnlyOne"
    board = niche_effect_board(data, build_knn_graph(data, k=8), min_ko=5, n_perm=0)
    assert board["rows"] == [] and "degenerate" in board.get("note", "")
