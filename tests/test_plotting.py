import numpy as np
from spbench.config import run_benchmark
from spbench.plotting import collect_prop_samples, collect_seed_samples, collect_niche_tier

GCN_KW = {"hidden": 16, "epochs": 5}


def test_collect_seed_samples(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    boxes, dashed = collect_seed_samples(res)
    assert len(boxes) >= 1
    assert "null (floor)" in dashed


import matplotlib
matplotlib.use("Agg")
from spbench.plotting import plot_seed_prop


def test_collect_niche_tier_base_has_cell1_upper(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    boxes, dashed = collect_niche_tier(res, "base")
    assert any("Gaussian" in k for k in boxes)          # seed-only + Gaussian box
    assert "null (floor)" in dashed
    assert "GT-seed (upper)" in dashed                  # cell-1 = GT+base
    import numpy as np
    exp = np.nanmean([c["e"]["GT+base"] for c in res["compare"].values()])
    assert np.isclose(dashed["GT-seed (upper)"], float(exp))
    assert "oracle" not in dashed


def test_collect_niche_tier_learned_has_cell2_upper(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    boxes, dashed = collect_niche_tier(res, "learned")
    assert any("GCN" in k for k in boxes)               # end-to-end mock box
    import numpy as np
    exp = np.nanmean([c["e"]["GT+learned"] for c in res["compare"].values()])
    assert np.isclose(dashed["GT-seed (upper)"], float(exp))   # cell-2 = GT+learned


def test_plot_seed_prop_three_boards(synth):
    import matplotlib; matplotlib.use("Agg")
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    fig = plot_seed_prop(res)
    assert len(fig.axes) == 3
    titles = [a.get_title() for a in fig.axes]
    assert any("seed" in t for t in titles)
    assert sum("niche" in t for t in titles) == 2
    seed_leg = fig.axes[0].get_legend()
    seed_labels = [t.get_text() for t in seed_leg.get_texts()] if seed_leg else []
    assert "GT-seed (upper)" not in seed_labels


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
