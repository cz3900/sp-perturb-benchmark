import numpy as np
from spbench.adapters.binan import _assemble_binan


def test_binan_assemble():
    X = np.arange(5 * 3).reshape(5, 3).astype(float)
    coords = np.arange(10).reshape(5, 2).astype(float)
    # one-hot guide table: control, guide_0, guide_1, multiplet(2 guides), control
    onehot = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 0]])
    data = _assemble_binan(X, coords, onehot, ["g0", "g1", "g2"], tumor_full_idx_1based=[2, 3])
    # sum 0 -> control, sum 1 -> guide_<argmax>, sum>=2 -> none
    assert list(data.perturbation) == ["control", "guide_0", "guide_1", "none", "control"]
    # cell_type: 1-based full index 2,3 -> tumor, else other
    assert list(data.cell_type) == ["other", "tumor", "tumor", "other", "other"]
    assert data.has_ntc is True
    assert set(data.perturbations()) == {"guide_0", "guide_1"}
    assert data.X.shape == (5, 3) and list(data.gene_names) == ["g0", "g1", "g2"]
    assert data.coords.shape == (5, 2)
