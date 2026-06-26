import numpy as np
import matplotlib
matplotlib.use("Agg")
from spbench.config import run_benchmark
from spbench.plotting import collect_delta, plot_delta, external_methods

GCN_KW = {"hidden": 16, "epochs": 5}


def test_collect_delta_seed_and_niche(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    sb, ref = collect_delta(res, "seed")
    nb, _ = collect_delta(res, "niche")
    assert ref == {"perfect": 1.0, "no-corr": 0.0}
    assert "seed model" in sb
    # niche board carries the deployable Gaussian/GCN PCC-delta boxes
    assert any("Gaussian" in k for k in nb)
    assert any("GCN" in k for k in nb)


def test_plot_delta_two_boards(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    fig = plot_delta(res)
    assert len(fig.axes) == 2
    titles = [a.get_title().lower() for a in fig.axes]
    assert any("seed" in t for t in titles) and any("niche" in t for t in titles)


def test_external_methods_empty_without_externals(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    assert external_methods(res) == []


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
