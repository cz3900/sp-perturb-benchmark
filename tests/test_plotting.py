import numpy as np
from spbench.config import run_benchmark
from spbench.plotting import collect_prop_samples, collect_seed_samples

GCN_KW = {"hidden": 16, "epochs": 5}


def test_collect_prop_samples_has_named_gcn(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    boxes, dashed = collect_prop_samples(res)
    assert "GCN" in boxes, "learned prop must show up named 'GCN'"
    assert "Gaussian" in boxes
    assert isinstance(boxes["GCN"], np.ndarray) and boxes["GCN"].size > 0
    assert "null" in dashed and "oracle" in dashed


def test_collect_seed_samples(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    boxes, dashed = collect_seed_samples(res)
    assert len(boxes) >= 1
    assert "null" in dashed


import matplotlib
matplotlib.use("Agg")
from spbench.plotting import plot_seed_prop


def test_plot_seed_prop_returns_two_axes_with_gcn(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    fig = plot_seed_prop(res)
    assert fig is not None
    assert len(fig.axes) == 2
    niche_labels = [t.get_text() for t in fig.axes[1].get_xticklabels()]
    assert "GCN" in niche_labels
    # seed (left) axis carries the seed-model box; niche (right) drew its dashed baselines (legend)
    assert "seed model" in [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert fig.axes[1].get_legend() is not None


def test_demo_script_writes_png(tmp_path):
    # smoke-test the actual demo SCRIPT (its imports + argparse + CLI), not a re-inlined body
    import subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    out = tmp_path / "methods.png"
    r = subprocess.run([sys.executable, str(root / "scripts" / "plot_methods_demo.py"),
                        "--out", str(out)], cwd=str(root), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 0
