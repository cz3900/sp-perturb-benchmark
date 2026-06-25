import os, importlib.util, numpy as np
_R = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "spatialprop", "run_spatialprop.py")

def _load():
    spec = importlib.util.spec_from_file_location("run_spatialprop", _R)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_build_multiplier_in_panel_gene():
    m = _load()
    genes = ["g1", "g2", "g3"]; celltypes = ["T", "B"]
    mult = m.build_multiplier_matrix("g2", genes, celltypes, value=0.0)
    # rows = celltypes, cols = genes; only the guide gene column is set (knockout -> 0.0), rest = 1.0
    assert mult.shape == (2, 3)
    assert np.allclose(mult[:, 1], 0.0)                         # g2 knocked out for all celltypes
    assert np.allclose(mult[:, [0, 2]], 1.0)

def test_build_multiplier_off_panel_gene_returns_none():
    m = _load()
    # guide gene not in the panel -> cannot be expressed as a multiplier input
    assert m.build_multiplier_matrix("gX", ["g1", "g2"], ["T"], value=0.0) is None
