"""Pure-logic tests for SPAC-seq prep (no real data): guide collapsing, aggregation, calling."""
import numpy as np
from scipy.sparse import csr_matrix

from spbench.adapters import spac_seq_prep as P
from spbench.data import CONTROL, UNLABELED


def test_guide_target_strips_sg_and_replicate():
    assert P.guide_target("sgTgfbr2_1") == "Tgfbr2"
    assert P.guide_target("sgTgfbr2_2") == "Tgfbr2"
    assert P.guide_target("sgH2-M10.1_2") == "H2-M10.1"   # dots/hyphens kept; only trailing _N dropped
    assert P.guide_target("sgnon-targeting_7") == P.NON_TARGETING
    assert P.guide_target("sgnon-targeting_50") == P.NON_TARGETING


def test_collapse_to_genes_sums_two_sgrnas():
    guides = ["sgA_1", "sgA_2", "sgB_1", "sgB_2", "sgnon-targeting_1"]
    # one cell: A_1=3, A_2=1, B_1=0, B_2=0, NT=2
    cg = csr_matrix(np.array([[3, 1, 0, 0, 2]], dtype=float))
    M, genes = P.collapse_to_genes(cg, guides)
    g2v = dict(zip(genes, M.toarray()[0]))
    assert g2v["A"] == 4 and g2v["B"] == 0 and g2v[P.NON_TARGETING] == 2
    assert genes[-1] == P.NON_TARGETING            # non-targeting sorted last


def test_call_perturbation_labels():
    genes = ["A", "B", P.NON_TARGETING]
    X = csr_matrix(np.array([
        [10, 0, 0],   # clean A KO
        [0,  1, 0],   # below min_umi -> none
        [4,  4, 0],   # ambiguous multiplet (dominance 1) -> none
        [0,  0, 6],   # non-targeting only -> control
    ], dtype=float))
    out = P.call_perturbation(X, genes, min_umi=2, min_dominance=2.0)
    assert list(out["perturbation"]) == ["A", UNLABELED, UNLABELED, CONTROL]


def test_aggregate_guides_to_cells_sums_assigned_bins():
    # 4 bins x 2 guides ; bins 0,1 -> cell 7 ; bin 2 -> cell 9 ; bin 3 unassigned (-1)
    bin_guide = csr_matrix(np.array([[1, 0], [2, 1], [0, 5], [9, 9]], dtype=float))
    bin_cellid = np.array([7, 7, 9, -1])
    cg = P.aggregate_guides_to_cells(bin_guide, bin_cellid, cell_id_order=np.array([7, 9]))
    assert np.array_equal(cg.toarray(), np.array([[3, 1], [0, 5]]))   # bin 3 dropped


def test_cellbarcode_to_id():
    assert P.cellbarcode_to_id("cellid_000000001-1") == 1
    assert P.cellbarcode_to_id("cellid_000000042-1") == 42


def test_assign_bins_to_cells_knn_gates_by_radius():
    # two unit-square cells centred at (0,0) and (10,0)
    polys = [np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float),
             np.array([[9, -1], [11, -1], [11, 1], [9, 1]], float)]
    cell_ids = np.array([7, 9])
    bins = np.array([[0.0, 0.0],     # -> cell 7
                     [10.0, 0.5],    # -> cell 9
                     [5.0, 0.0]])    # midpoint, beyond both radii -> -1
    out, off = P.assign_bins_to_cells(bins, cell_ids, polys, offset=np.zeros(2), method="knn")
    assert out.tolist() == [7, 9, -1]
