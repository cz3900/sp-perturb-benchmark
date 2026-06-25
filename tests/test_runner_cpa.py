import os, importlib.util, numpy as np
_R = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "cpa", "run_cpa.py")

def _load():
    spec = importlib.util.spec_from_file_location("run_cpa", _R)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m   # no cpa import

def test_celltype_profiles_from_pred_groups_by_cell_type():
    m = _load()
    # 4 predicted control rows, cell types [T,T,B,B], CPA_pred values per row
    pred = np.array([[1., 1.], [3., 3.], [10., 10.], [12., 12.]])
    cts = np.array(["T", "T", "B", "B"])
    prof = m.celltype_profiles_from_pred(pred, cts)
    assert np.allclose(prof["T"], [2., 2.])      # mean of the two T rows
    assert np.allclose(prof["B"], [11., 11.])
    assert None in prof                          # global fallback present

def test_build_seed_uses_shared_aggregate(tmp_path):
    m = _load()
    prof = {"T": np.array([2., 2.]), "B": np.array([11., 11.]), None: np.array([6., 6.])}
    seed = m.build_seed(np.array(["B", "T"]), prof)
    assert np.allclose(seed, [[11., 11.], [2., 2.]])
