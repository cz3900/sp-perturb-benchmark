import matplotlib
matplotlib.use("Agg")
from spbench.viz import (plot_2x2, plot_attribution, plot_learned_value,
                         plot_significance_contrast, plot_slope, plot_seed_vs_learned)
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
