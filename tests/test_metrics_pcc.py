import numpy as np
from spbench.metrics import get_metric, list_metrics

def test_pcc_and_mse_registered():
    # the pluggable registry must expose the new primitives like every other metric
    for name in ("pcc_delta", "mse"):
        assert name in list_metrics()
        assert get_metric(name).compute is not None

def test_pcc_delta_direction():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(60, 8))
    shift = np.zeros(8); shift[2] = 3.0; shift[5] = -2.0
    pred = ref + shift + rng.normal(scale=0.1, size=(60, 8))
    gt = ref + shift + rng.normal(scale=0.1, size=(60, 8))
    m = get_metric("pcc_delta")
    assert m.compute(pred, gt, {"reference": ref}) > 0.9          # same shift -> ~1
    assert m.compute(ref - shift, gt, {"reference": ref}) < 0     # opposite shift -> negative

def test_mse_zero_when_equal_and_higher_is_better_flags():
    x = np.ones((10, 5))
    assert get_metric("mse").compute(x, x) == 0.0
    assert get_metric("mse").higher_is_better is False
    assert get_metric("pcc_delta").higher_is_better is True
