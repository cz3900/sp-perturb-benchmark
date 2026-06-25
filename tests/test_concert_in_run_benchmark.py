import numpy as np, h5py
from spbench.synthetic import make_synthetic
from spbench.config import run_benchmark
from spbench.models.concert_model import ConcertModel


def test_concert_scores_as_external_in_summary(tmp_path):
    data = make_synthetic(0)
    P = data.perturbations()[0]
    # mock CONCERT counterfactual: predicted expression for EVERY cell (N, G)
    p = tmp_path / f"{P}.h5ad"
    with h5py.File(p, "w") as f:
        f.create_dataset("X", data=data.X + 0.1)
    model = ConcertModel({P: str(p)})
    res = run_benchmark(data, perturbations=[P], external_models={"CONCERT": model}, progress=False)
    # CONCERT appears as an external niche method with a (finite or NaN) PCC-delta
    assert "CONCERT" in res["compare"][P]["pcc"]
    from spbench.plotting import summary_table
    row = summary_table(res)[0]
    assert "niche_CONCERT" in row
