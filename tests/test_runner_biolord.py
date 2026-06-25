import os, importlib.util, numpy as np
_R = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "biolord", "run_biolord.py")

def _load():
    spec = importlib.util.spec_from_file_location("run_biolord", _R)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_build_seed_maps_celltype_profiles():
    m = _load()
    prof = {"T": np.array([2., 2.]), "B": np.array([9., 9.]), None: np.array([5., 5.])}
    seed = m.build_seed(np.array(["T", "B", "Z"]), prof)     # Z -> global fallback
    assert np.allclose(seed, [[2., 2.], [9., 9.], [5., 5.]])
