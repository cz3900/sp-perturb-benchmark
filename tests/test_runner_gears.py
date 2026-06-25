import os, importlib.util, numpy as np
_R = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "gears", "run_gears.py")

def _load():
    spec = importlib.util.spec_from_file_location("run_gears", _R)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_broadcast_seed_tiles_single_mean_to_centers():
    m = _load()
    pred_mean = np.array([1., 2., 3.])
    seed = m.broadcast_seed(pred_mean, 4)
    assert seed.shape == (4, 3)
    assert np.allclose(seed, np.tile(pred_mean, (4, 1)))
