import numpy as np, pytest
from spbench.dump_align import align_prediction_to_panel

def test_align_fills_unpredicted_from_fallback():
    pred = np.array([[10., 20.]])              # model predicted genes b, d only
    fallback = np.array([[1., 2., 3., 4.]])    # panel order a, b, c, d (unperturbed input)
    out = align_prediction_to_panel(pred, ["b", "d"], ["a", "b", "c", "d"], fallback)
    assert np.allclose(out, [[1., 10., 3., 20.]])

def test_align_row_mismatch_raises():
    with pytest.raises(ValueError):
        align_prediction_to_panel(np.zeros((2, 1)), ["a"], ["a", "b"], np.zeros((3, 2)))
