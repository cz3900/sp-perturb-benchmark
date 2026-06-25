import numpy as np
from spbench.aggregate import normalized_pcc
from spbench.compare import _mag


def test_normalized_pcc_anchors_and_clip():
    assert normalized_pcc(0.0, 0.0, 0.8) == 0.0          # at null -> 0
    assert normalized_pcc(0.8, 0.0, 0.8) == 1.0          # at upper -> 1
    assert abs(normalized_pcc(0.4, 0.0, 0.8) - 0.5) < 1e-9
    assert normalized_pcc(1.0, 0.0, 0.8) == 1.0          # above upper -> clipped to 1
    assert normalized_pcc(-0.3, 0.0, 0.8) == 0.0         # below null -> clipped to 0
    assert np.isnan(normalized_pcc(0.4, 0.5, 0.5))       # zero null/upper gap -> nan
    assert np.isnan(normalized_pcc(-0.05, 0.0, -0.02))   # upper below null (no headroom) -> nan


def test_mag_ratio():
    ref = np.zeros((20, 4))
    obs = np.full((20, 4), 2.0)                          # true mean-shift = (2,2,2,2)
    assert abs(_mag(np.full((20, 4), 2.0), obs, ref) - 1.0) < 1e-9   # exact -> 1
    assert abs(_mag(np.full((20, 4), 1.0), obs, ref) - 0.5) < 1e-9   # half  -> 0.5
    assert abs(_mag(np.full((20, 4), 4.0), obs, ref) - 2.0) < 1e-9   # double -> 2 (over-shoot)
    assert np.isnan(_mag(obs, ref, ref))                # no true shift -> nan


def test_run_benchmark_exposes_mag(synth):
    from spbench.config import run_benchmark
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs={"hidden": 16, "epochs": 5}, progress=False)
    c = res["compare"]["P0"]; s = res["seed"]["P0"]
    assert "mag" in c and "model+learned" in c["mag"] and np.isfinite(c["mag"]["model+learned"])
    assert "mag" in s
