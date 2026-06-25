import numpy as np, h5py
from spbench.models.seed_dump import SeedDumpModel

def _write_dump(path, seed_pred, centers):
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=np.asarray(seed_pred, float))
        g = f.create_group("obs"); g.create_dataset("center_idx", data=np.asarray(centers, np.int64))

def test_seed_dump_serves_aligned_array(tmp_path):
    p = tmp_path / "GeneA_seed.h5ad"
    _write_dump(str(p), [[1., 2.], [3., 4.]], [0, 5])
    m = SeedDumpModel("cpa", {"GeneA": str(p)}).fit(None)
    assert m.name == "cpa"
    out = m.predict_seed("GeneA", np.zeros((9, 2)))
    assert np.allclose(out, [[1., 2.], [3., 4.]])
    assert list(m.centers("GeneA")) == [0, 5]

def test_seed_dump_caches(tmp_path):
    p = tmp_path / "G_seed.h5ad"; _write_dump(str(p), [[1., 2.]], [0])
    m = SeedDumpModel("gears", {"G": str(p)})
    assert m.predict_seed("G", np.zeros((1, 2))) is m.predict_seed("G", np.zeros((1, 2)))

def test_scgen_still_subclasses_and_works(tmp_path):
    from spbench.models.scgen_model import ScgenSeedModel
    p = tmp_path / "GeneB_seed.h5ad"; _write_dump(str(p), [[7., 8.]], [2])
    m = ScgenSeedModel({"GeneB": str(p)})
    assert m.name == "scgen"
    assert np.allclose(m.predict_seed("GeneB", np.zeros((1, 2))), [[7., 8.]])
    assert list(m.centers("GeneB")) == [2]
