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
from spbench.reference_aggregate import control_reference_centers


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
    refs = control_reference_centers(data, centers)        # fill_2x2's aggregate-control reference     (all same-type controls, no longer k-matched)
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


# --- G6.4: offline runner pure functions (loaded by path; must NOT import scgen) ---
import importlib.util
import os

_RUNNER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "scgen", "run_scgen.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_scgen", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                          # must NOT import scgen at module level
    return mod


def test_aggregate_control_predict_maps_celltype_means():
    runner = _load_runner()
    # 3 centers, cell types [T, B, T]; per-cell-type predicted profile bank
    center_cell_types = np.array(["T", "B", "T"])
    ct_profiles = {"T": np.array([1.0, 1.0, 1.0]), "B": np.array([9.0, 9.0, 9.0])}
    seed_pred = runner.aggregate_control_predict(center_cell_types, ct_profiles)
    assert seed_pred.shape == (3, 3)
    assert np.allclose(seed_pred[0], [1.0, 1.0, 1.0])    # T center -> T aggregate profile
    assert np.allclose(seed_pred[1], [9.0, 9.0, 9.0])    # B center -> B aggregate profile
    assert np.allclose(seed_pred[2], [1.0, 1.0, 1.0])    # second T center -> same T profile


def test_aggregate_control_predict_falls_back_to_global_for_missing_ct():
    runner = _load_runner()
    center_cell_types = np.array(["T", "Z"])             # Z has no per-ct profile
    ct_profiles = {"T": np.array([2.0, 2.0]), None: np.array([5.0, 5.0])}
    seed_pred = runner.aggregate_control_predict(center_cell_types, ct_profiles)
    assert np.allclose(seed_pred[0], [2.0, 2.0])
    assert np.allclose(seed_pred[1], [5.0, 5.0])         # global (None) fallback


def test_write_seed_dump_roundtrips_for_loader(tmp_path):
    runner = _load_runner()
    seed_pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    centers = np.array([0, 3])
    p = tmp_path / "GeneA_seed.h5ad"
    runner.write_seed_dump(str(p), seed_pred, centers)
    # the loader from G6.2 must read it back identically
    from spbench.models.scgen_model import ScgenSeedModel
    model = ScgenSeedModel({"GeneA": str(p)})
    assert np.allclose(model.predict_seed("GeneA", np.zeros((2, 3))), seed_pred)
    assert list(model.centers("GeneA")) == [0, 3]


# --- G6.5: end-to-end integration (export -> mock seed dump -> loader -> fill_2x2(eval_X) ->
#           evaluate_seed), threading G6.1-G6.4 together WITHOUT the scgen conda env. ---


def _counts_data():
    rng = np.random.default_rng(3)
    n, g = 40, 6
    counts = rng.integers(0, 20, size=(n, g)).astype(float)
    coords = rng.uniform(0, 100, size=(n, 2))
    pert = np.array([UNLABELED] * n, dtype=object)
    pert[:6] = "GeneA"; pert[6:18] = CONTROL                # 6 perturbed centers, 12 controls
    cell_type = np.where(np.arange(n) % 2 == 0, "T", "B")
    return StandardData(
        X=rng.normal(size=(n, g)),                          # z-scored-like; eval_X overrides it
        coords=coords, perturbation=pert.astype(str),
        cell_type=cell_type.astype(str), batch=np.array(["s1"] * n),
        gene_names=[f"g{i}" for i in range(g)], meta={"counts": counts})


def test_scgen_end_to_end_export_load_score(tmp_path):
    data = _counts_data()
    edges = build_knn_graph(data, k=6)
    P = "GeneA"
    lognorm_X = build_lognorm_X(data)

    # 1. export adapter (G6.1): integer counts -> log-norm 2-condition AnnData
    info = export_to_scgen_h5(data, P, lognorm_X, str(tmp_path / f"{P}.h5ad"))
    assert info["n_stim"] == 6 and info["n_ctrl"] == 12

    # 2. mock offline runner output (G6.4 write_seed_dump, via the G6.2 test helper):
    #    a "perfect" aligned seed_pred = the observed centers in log-norm space.
    centers = np.where(data.perturbation == P)[0]
    seed_dump = tmp_path / f"{P}_seed.h5ad"
    _write_seed_dump(str(seed_dump), lognorm_X[centers], centers)

    # 3. loader (G6.2) -> fill_2x2 with eval_X = lognorm_X (G6.3)
    model = ScgenSeedModel({P: str(seed_dump)}).fit(None)
    prop = _IdentityProp()
    grid = fill_2x2(data, P, edges, model, prop, prop, return_niches=True, eval_X=lognorm_X)
    n = grid["_niches"]
    assert n["seed_pred"].shape == (len(centers), data.n_genes)
    assert np.allclose(n["seed_obs"], lognorm_X[centers])          # obs in eval_X (log-norm) space
    refs = control_reference_centers(data, centers)        # fill_2x2's aggregate-control reference             (all same-type controls, no longer k-matched)
    seed_ref_idx = np.unique(np.concatenate(refs))
    assert np.allclose(n["seed_ref"], lognorm_X[seed_ref_idx])     # ref in eval_X space

    # 4. evaluate_seed: finite mse + well-defined pcc in the unified log-norm space
    res = evaluate_seed(n)
    assert np.isfinite(res["mse"]) and res["mse"] >= 0
    assert res["n"] == len(centers)
    assert np.isfinite(res["pcc_delta"]) or np.isnan(res["pcc_delta"])


def test_scgen_per_center_seed_survives_fill_2x2(tmp_path):
    """Regression for design must-fix #2: ScgenSeedModel's cached (n_centers, G) array is per-center
    aligned (via /obs/center_idx). fill_2x2 must propagate EACH center's OWN cached row as that
    center's seed, NOT collapse the whole array to one global-mean vector broadcast everywhere
    (which is what the per-`rc` `predict_seed(...).mean(0)` path did). With DISTINCT per-center rows,
    a collapse makes every seed_pred row identical; the per-center alignment must keep them distinct.
    """
    data = _counts_data()
    edges = build_knn_graph(data, k=6)
    P = "GeneA"
    centers = np.where(data.perturbation == P)[0]
    G = data.n_genes

    # DISTINCT per-center rows: row for center index c = a vector unique to c (NOT all equal).
    # Dump rows are written in a SHUFFLED center order to prove alignment goes through
    # /obs/center_idx, not positional luck (np.where gives ascending order).
    dump_centers = centers[::-1].copy()                       # reversed -> != np.where order
    dump_rows = np.array([np.full(G, float(c + 1)) for c in dump_centers])   # row value encodes its center
    seed_dump = tmp_path / f"{P}_seed.h5ad"
    _write_seed_dump(str(seed_dump), dump_rows, dump_centers)

    model = ScgenSeedModel({P: str(seed_dump)}).fit(None)
    prop = _IdentityProp()
    grid = fill_2x2(data, P, edges, model, prop, prop, return_niches=True)
    seed_pred = grid["_niches"]["seed_pred"]

    # Expected: seed_pred row i (for center centers[i]) == that center's own cached row, which by
    # construction is full(G, centers[i] + 1). Aligned to np.where(data.perturbation==P)[0] order.
    expected = np.array([np.full(G, float(c + 1)) for c in centers])
    assert seed_pred.shape == (len(centers), G)
    assert np.allclose(seed_pred, expected), "per-center cached rows must survive into seed_pred"

    # The collapse signature: with the old global-mean path every row is identical. The distinct
    # per-center rows here have different per-center values, so they MUST NOT all be equal.
    assert not np.allclose(seed_pred, seed_pred[0]), "seed_pred collapsed to one broadcast vector"
