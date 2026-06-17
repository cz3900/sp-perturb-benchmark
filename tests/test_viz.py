import matplotlib
matplotlib.use("Agg")
from spbench.viz import (plot_2x2, plot_attribution, plot_learned_value,
                         plot_significance_contrast, plot_slope, plot_seed_vs_learned,
                         plot_baseline_gain, plot_gain_per_perturbation, plot_aggregate_2x2)
from spbench.config import run_benchmark
from spbench.synthetic import make_synthetic

def test_plots_return_figures():
    res = run_benchmark(make_synthetic(0), perturbations=["P0"], k=8,
                        gcn_kwargs={"hidden": 16, "epochs": 3})
    fig1 = plot_2x2(res["grids"]["P0"], title="P0")
    fig2 = plot_attribution(res["attribution"])
    assert fig1 is not None and fig2 is not None

def test_result_figures_return_figures():
    # multi-perturbation result with a significant / non-significant split
    res = run_benchmark(make_synthetic(0), perturbations=["P0", "P1", "P2"], k=8,
                        gcn_kwargs={"hidden": 16, "epochs": 3})
    significant = ["P0", "P1"]
    assert plot_learned_value(res, significant) is not None
    assert plot_significance_contrast(res, significant) is not None
    assert plot_slope(res, significant) is not None
    assert plot_seed_vs_learned(res, significant) is not None

def test_baseline_gain_figures():
    # compare=True attaches per-perturbation energy distances + gains over the no-effect baseline
    res = run_benchmark(make_synthetic(0), perturbations=["P0", "P1", "P2"], k=8,
                        gcn_kwargs={"hidden": 16, "epochs": 5}, compare=True, progress=False)
    assert "compare" in res
    c = res["compare"]["P0"]
    assert "gain" in c and "model+learned" in c["gain"]
    assert c["gain"]["null"] == 0.0                       # baseline gain is 0 by construction
    assert plot_baseline_gain(res) is not None
    assert plot_gain_per_perturbation(res, ["P0", "P1"]) is not None
    assert plot_aggregate_2x2(res) is not None
