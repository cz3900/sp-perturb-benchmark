import numpy as np
import h5py
import pytest
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.concert_export import export_to_concert_h5
from spbench.models.concert_model import ConcertModel, read_h5ad_X
from spbench.synthetic import make_synthetic
from spbench.graph import build_knn_graph, neighbors_of
from spbench.compare import compare_to_baseline


def _toy():
    return StandardData(
        X=np.array([[1, 2, 0], [0, 3, 1], [2, 0, 4], [1, 1, 1]], float),
        coords=np.array([[0, 0], [10, 0], [0, 10], [10, 10]], float),
        perturbation=np.array(["GeneA", CONTROL, UNLABELED, "GeneA"]),
        cell_type=np.array(["T", "T", "B", "B"]),
        batch=np.array(["s1"] * 4), gene_names=["g1", "g2", "g3"])


def test_export_to_concert_h5(tmp_path):
    p = tmp_path / "concert.h5"
    counts = export_to_concert_h5(_toy(), str(p))
    assert counts["GeneA"] == 2 and counts["None"] == 2          # control + no-guide -> 'None'
    with h5py.File(p, "r") as f:
        assert f["X"].shape == (4, 3)
        assert f["pos"].shape == (2, 4)                          # transposed
        assert list(np.array(f["perturbation"]).astype(str)) == ["GeneA", "None", "None", "GeneA"]
        assert list(np.array(f["tissue"]).astype(str)) == ["T", "T", "B", "B"]
        assert list(np.array(f["gene"]).astype(str)) == ["g1", "g2", "g3"]


def test_export_rejects_non_integer_counts(tmp_path):
    d = _toy(); d.X = d.X + 0.5                                   # normalised/scaled -> not counts
    with pytest.raises(ValueError):
        export_to_concert_h5(d, str(tmp_path / "bad.h5"))


def test_concert_model_extracts_bystander_niche(tmp_path):
    data = make_synthetic(0)
    edges = build_knn_graph(data, k=8)
    # mock CONCERT counterfactual .h5ad: row i has the constant value i (so we can identify rows)
    Xpred = np.tile(np.arange(data.n_cells)[:, None], (1, data.n_genes)).astype(float)
    p = tmp_path / "P0_pred.h5ad"
    with h5py.File(p, "w") as f:
        f.create_dataset("X", data=Xpred)
    assert read_h5ad_X(str(p)).shape == (data.n_cells, data.n_genes)

    model = ConcertModel({"P0": str(p)}).fit(data, edges)
    niche = model.predict_niche(data, "P0", edges)

    # independently compute the expected bystander indices
    exp = []
    for c in np.where(data.perturbation == "P0")[0]:
        nb = neighbors_of(c, edges)
        exp.append(nb[~data.is_perturbed[nb]])
    exp = np.concatenate(exp)
    assert niche.shape == (len(exp), data.n_genes)
    assert np.allclose(niche[:, 0], exp)                         # row i carried value i


def test_compare_accepts_extra_concert_cloud():
    rng = np.random.default_rng(0)
    g = 8
    niches = {"observed": rng.normal(size=(60, g)), "reference": rng.normal(size=(80, g))}
    for k in ("1", "2", "3", "4"):
        niches[k] = rng.normal(size=(60, g))
    concert = rng.normal(size=(60, g))
    res = compare_to_baseline(niches, extra={"CONCERT": concert})
    for field in ("pcc", "mag"):
        assert "CONCERT" in res[field]
    assert "null" in res["pcc"]
