import numpy as np
import h5py
import pytest
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.scgen_export import build_lognorm_X, export_to_scgen_h5


def _toy():
    # 4 cells: GeneA(stim), control, none(unlabeled), GeneA(stim); integer counts in meta['counts']
    counts = np.array([[2, 4, 0], [0, 6, 2], [4, 0, 8], [2, 2, 2]], float)
    d = StandardData(
        X=np.zeros((4, 3), float),                       # X is z-scored / irrelevant here
        coords=np.array([[0, 0], [10, 0], [0, 10], [10, 10]], float),
        perturbation=np.array(["GeneA", CONTROL, UNLABELED, "GeneA"]),
        cell_type=np.array(["T", "T", "B", "B"]),
        batch=np.array(["s1"] * 4), gene_names=["g1", "g2", "g3"],
        meta={"counts": counts})
    return d


def test_build_lognorm_X_matches_recipe():
    d = _toy()
    X = build_lognorm_X(d)
    counts = d.meta["counts"]
    expected = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
    assert X.shape == (4, 3)
    assert np.allclose(X, expected)
    assert np.isfinite(X).all()


def test_build_lognorm_X_rejects_non_integer_counts():
    d = _toy()
    d.meta["counts"] = d.meta["counts"] + 0.5            # pre-normalized -> not raw integer counts
    with pytest.raises(ValueError):
        build_lognorm_X(d)


def test_export_to_scgen_h5_condition_map_and_drop(tmp_path):
    d = _toy()
    lognorm_X = build_lognorm_X(d)
    p = tmp_path / "GeneA.h5ad"
    info = export_to_scgen_h5(d, "GeneA", lognorm_X, str(p))
    assert info["n_stim"] == 2 and info["n_ctrl"] == 1
    with h5py.File(p, "r") as f:
        # kept = 2 GeneA (stim) + 1 control; 'none' dropped
        assert f["X"].shape == (3, 3)
        cond = list(np.array(f["obs"]["condition"]).astype(str))
        assert cond == ["stimulated", "control", "stimulated"]
        ct = list(np.array(f["obs"]["cell_type"]).astype(str))
        assert ct == ["T", "T", "B"]
        oi = list(np.array(f["obs"]["orig_idx"]))
        assert oi == [0, 1, 3]                            # orig cell ids of kept rows
        assert list(np.array(f["var"]["gene_names"]).astype(str)) == ["g1", "g2", "g3"]
        # normalization recipe stored for cross-env eval_X reproducibility
        assert float(f["uns"].attrs["target_sum"]) == 1e4
        assert int(f["uns"].attrs["log1p"]) == 1
        assert np.allclose(np.array(f["X"]), lognorm_X[[0, 1, 3]])


from spbench.models.scgen_model import ScgenSeedModel
from spbench.models.concert_model import read_h5ad_X


def _write_seed_dump(path, seed_pred, centers):
    """Mock of run_scgen.write_seed_dump (G6.4): /X aligned to centers order + /obs/center_idx."""
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=np.asarray(seed_pred, float))
        g_obs = f.create_group("obs")
        g_obs.create_dataset("center_idx", data=np.asarray(centers, np.int64))


def test_scgen_loader_serves_aligned_seed_array(tmp_path):
    # offline runner output: 2 centers, 3 genes; rows already in centers order
    seed_pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    centers = [0, 3]
    p = tmp_path / "GeneA_seed.h5ad"
    _write_seed_dump(str(p), seed_pred, centers)

    model = ScgenSeedModel({"GeneA": str(p)}).fit(None)
    # ABC signature: predict_seed(perturbation, reference_cells) -> the cached aligned array.
    # reference_cells is ignored for the array's value (it is the offline-aligned seed_pred).
    out = model.predict_seed("GeneA", np.zeros((5, 3)))
    assert out.shape == (2, 3)
    assert np.allclose(out, seed_pred)
    assert np.allclose(read_h5ad_X(str(p)), seed_pred)   # reuses concert read_h5ad_X


def test_scgen_loader_centers_accessor(tmp_path):
    seed_pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    centers = [2, 7]
    p = tmp_path / "GeneB_seed.h5ad"
    _write_seed_dump(str(p), seed_pred, centers)
    model = ScgenSeedModel({"GeneB": str(p)})
    assert list(model.centers("GeneB")) == [2, 7]        # for fill_2x2 to align seed_obs/seed_ref


def test_scgen_loader_caches_per_perturbation(tmp_path):
    seed_pred = np.array([[1.0, 2.0]])
    p = tmp_path / "G_seed.h5ad"
    _write_seed_dump(str(p), seed_pred, [0])
    model = ScgenSeedModel({"G": str(p)})
    a = model.predict_seed("G", np.zeros((1, 2)))
    b = model.predict_seed("G", np.zeros((1, 2)))
    assert a is b                                        # second call hits the cache (same object)


# --- G6.3: eval_X as a precomputed matrix switches seed_obs/seed_ref to the log-norm space ---
from spbench.harness import fill_2x2
from spbench.compare import evaluate_seed
from spbench.synthetic import make_synthetic
from spbench.graph import build_knn_graph
from spbench.reference import match_reference_centers


class _ConstSeed:
    name = "const_seed"
    def __init__(self, val): self.val = val
    def fit(self, train): return self
    def predict_seed(self, perturbation, reference_cells):
        return np.tile(self.val, (max(1, len(reference_cells)), 1)).astype(float)


class _IdentityProp:
    name = "id_prop"
    def fit(self, train, edges): return self
    def propagate(self, X_reference, edges, center, seed_state, neighbors):
        return X_reference[neighbors]


def test_fill_2x2_eval_X_switches_seed_obs_ref_space():
    data = make_synthetic(0)
    edges = build_knn_graph(data, k=8)
    P = data.perturbations()[0]
    eval_X = data.X + 100.0                                # a distinct "log-norm-like" space
    seed = _ConstSeed(np.zeros(data.n_genes)); prop = _IdentityProp()
    grid = fill_2x2(data, P, edges, seed, prop, prop, return_niches=True, eval_X=eval_X)
    n = grid["_niches"]

    centers = np.where(data.perturbation == P)[0]
    refs = match_reference_centers(data, centers, k=5)     # k=5 == fill_2x2's default k_ref
    seed_ref_idx = np.unique(np.concatenate(refs))
    assert np.allclose(n["seed_obs"], eval_X[centers])     # observed centers in eval_X space
    assert np.allclose(n["seed_ref"], eval_X[seed_ref_idx])
    assert n["seed_pred"].shape == (len(centers), data.n_genes)


def test_fill_2x2_default_keeps_data_X_space():
    data = make_synthetic(0)
    edges = build_knn_graph(data, k=8)
    P = data.perturbations()[0]
    seed = _ConstSeed(np.zeros(data.n_genes)); prop = _IdentityProp()
    grid = fill_2x2(data, P, edges, seed, prop, prop, return_niches=True)   # no eval_X
    centers = np.where(data.perturbation == P)[0]
    assert np.allclose(grid["_niches"]["seed_obs"], data.X[centers])        # unchanged default


def test_evaluate_seed_finite_under_eval_X():
    data = make_synthetic(0)
    edges = build_knn_graph(data, k=8)
    P = data.perturbations()[0]
    centers = np.where(data.perturbation == P)[0]
    eval_X = np.abs(data.X) + 0.5                          # strictly positive, distinct space

    # a seed that predicts a real (non-flat) shift so pcc_delta is well-defined, not NaN-flat
    class _ShiftSeed:
        name = "shift"
        def fit(self, train): return self
        def predict_seed(self, perturbation, reference_cells):
            return eval_X[centers]                         # (n_centers, G); fill_2x2 .mean(0)s it
    prop = _IdentityProp()
    grid = fill_2x2(data, P, edges, _ShiftSeed(), prop, prop, return_niches=True, eval_X=eval_X)
    res = evaluate_seed(grid["_niches"])
    assert np.isfinite(res["mse"])
    assert res["n"] == int(len(centers))
