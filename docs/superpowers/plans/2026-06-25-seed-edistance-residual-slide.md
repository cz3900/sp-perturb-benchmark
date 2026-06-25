# Seed E-Distance Residual Slide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one PPT-ready mechanism figure explaining why mean-field seed predictions inflate E-distance and how same-cell-type control residuals fix the comparison.

**Architecture:** Add a focused Matplotlib script that creates a deterministic three-panel figure: observed seed cloud, collapsed mean-field prediction, and residual-restored prediction. The script exports both PNG and SVG for easy insertion into slides, and tests cover the figure structure plus CLI export without relying on real data.

**Tech Stack:** Python 3.10+, matplotlib Agg backend, numpy, pytest.

---

## File Structure

- Create `scripts/plot_seed_edistance_residual_slide.py`
  - Responsibility: generate the single mechanism figure and export it from a CLI.
  - Public functions: `make_figure()` and `main(argv=None)`.
  - No dependency on benchmark data files; the figure uses deterministic illustrative points.
- Create `tests/test_seed_edistance_residual_slide.py`
  - Responsibility: smoke-test `make_figure()` labels and CLI output.
- Generated outputs during manual use:
  - `notebooks/seed_edistance_residual_mechanism.png`
  - `notebooks/seed_edistance_residual_mechanism.svg`

## Task 1: Add the Figure Generator

**Files:**
- Create: `scripts/plot_seed_edistance_residual_slide.py`
- Test: not yet

- [ ] **Step 1: Create the script file**

Use `apply_patch` to create `scripts/plot_seed_edistance_residual_slide.py` with this complete content:

```python
"""Generate a PPT-ready mechanism figure for seed E-distance variance collapse.

The figure explains a metric artifact: mean-field seed predictions can collapse to one point,
making E-distance artificially high because the prediction has no within-cloud spread. Adding
same-cell-type control residuals restores realistic variance while preserving the mean.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GREEN = "#239b6b"
ORANGE = "#e6862d"
RED = "#d84a4a"
GRAY = "#667085"
DARK = "#1f2937"
LIGHT_GRAY = "#f3f5f7"


def _ellipse(ax, center, width, height, color):
    from matplotlib.patches import Ellipse

    patch = Ellipse(
        center,
        width,
        height,
        facecolor="none",
        edgecolor=color,
        linewidth=2.0,
        linestyle=(0, (2, 3)),
        alpha=0.95,
    )
    ax.add_patch(patch)


def _panel_title(ax, title, subtitle):
    ax.text(0.02, 0.95, title, transform=ax.transAxes, fontsize=16, fontweight="bold",
            color=DARK, va="top")
    ax.text(0.02, 0.87, subtitle, transform=ax.transAxes, fontsize=10.5,
            color=GRAY, va="top")


def _badge(ax, x, y, text, color, face):
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
        color=color, va="center", ha="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=face, edgecolor=color, linewidth=1.0),
    )


def _arrow_between(fig, ax0, ax1, text):
    b0 = ax0.get_position()
    b1 = ax1.get_position()
    y = (b0.y0 + b0.y1) / 2
    x0 = b0.x1 + 0.008
    x1 = b1.x0 - 0.008
    ax0.annotate(
        "",
        xy=(x1, y), xycoords=fig.transFigure,
        xytext=(x0, y), textcoords=fig.transFigure,
        arrowprops=dict(arrowstyle="-|>", color=GRAY, linewidth=1.8),
        annotation_clip=False,
    )
    fig.text((x0 + x1) / 2, y + 0.045, text, ha="center", va="bottom", fontsize=10.5, color=GRAY)


def make_figure(figsize=(13.4, 7.2)):
    """Return a deterministic three-panel mechanism figure.

    The points are illustrative, not computed from a dataset. They intentionally use `N` instead
    of a fixed count so the slide explains the general mechanism rather than one slice-specific
    example.
    """
    obs = np.array([
        [-0.45, 0.30], [-0.20, 0.55], [0.15, 0.40], [0.38, 0.18],
        [-0.55, -0.10], [-0.22, -0.26], [0.08, -0.40], [0.46, -0.18],
        [-0.05, 0.02], [0.24, -0.05],
    ])
    resid = np.array([
        [-0.35, 0.28], [-0.08, 0.48], [0.28, 0.30], [0.45, -0.02],
        [-0.43, -0.28], [-0.10, -0.38], [0.20, -0.25], [0.05, 0.05],
    ])

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.0, 1.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("white")

    ax = axes[0]
    _panel_title(ax, "Observed perturbed seeds", "target is a real cell cloud")
    _ellipse(ax, (0.0, 0.02), 1.45, 1.15, GREEN)
    ax.scatter(obs[:, 0], obs[:, 1], s=80, color=GREEN, zorder=3)
    _badge(ax, 0.50, 0.15, "N perturbed cells", GREEN, "#ecf8f2")
    ax.text(0.50, 0.06, "real cell-to-cell variance", transform=ax.transAxes,
            ha="center", va="center", fontsize=10.5, color=GRAY)

    ax = axes[1]
    _panel_title(ax, "Mean-field seed prediction", "same cell type -> same vector")
    ax.scatter([0], [0.04], s=230, color=ORANGE, zorder=4)
    for r in np.linspace(0.06, 0.22, 4):
        ax.scatter([0], [0.04], s=230 + r * 900, facecolors="none", edgecolors=ORANGE,
                   alpha=0.18, linewidths=1.3)
    _badge(ax, 0.50, 0.15, "collapsed cloud", RED, "#fff1f1")
    ax.text(0.50, 0.06, "spread(pred) near zero", transform=ax.transAxes,
            ha="center", va="center", fontsize=10.5, color=RED, fontweight="bold")

    ax = axes[2]
    _panel_title(ax, "Prediction + control residual", "variance restored, mean preserved")
    _ellipse(ax, (0.0, 0.02), 1.30, 1.05, ORANGE)
    ax.scatter(resid[:, 0], resid[:, 1], s=80, color=ORANGE, zorder=3)
    ax.scatter([0.0], [0.04], s=95, color="white", edgecolor=DARK, linewidth=1.4, zorder=5)
    ax.plot([-0.08, 0.08], [0.04, 0.04], color=DARK, linewidth=1.2, zorder=6)
    ax.plot([0.0, 0.0], [-0.04, 0.12], color=DARK, linewidth=1.2, zorder=6)
    _badge(ax, 0.50, 0.15, "fair E-distance", GREEN, "#ecf8f2")
    ax.text(0.50, 0.06, "mean unchanged; spread matched", transform=ax.transAxes,
            ha="center", va="center", fontsize=10.5, color=GRAY)

    fig.subplots_adjust(left=0.045, right=0.985, top=0.84, bottom=0.25, wspace=0.18)
    _arrow_between(fig, axes[0], axes[1], "E compares distributions")
    _arrow_between(fig, axes[1], axes[2], "+ same-type control residual")

    fig.text(
        0.05, 0.155,
        "Energy distance:  E = 2.cross(pred, obs) - spread(pred) - spread(obs)",
        ha="left", va="center", fontsize=14.0, color=DARK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor=LIGHT_GRAY, edgecolor="none"),
    )
    fig.text(
        0.05, 0.075,
        "Problem: if prediction is a point cloud, spread(pred) is near zero, so E can look worse than null even when the mean shift is useful.",
        ha="left", va="center", fontsize=11.2, color=GRAY,
    )
    fig.text(
        0.05, 0.035,
        "Fix: use residual-restored predictions for E-distance; keep raw mean predictions for PCC-delta and MSE.",
        ha="left", va="center", fontsize=11.2, color=GRAY,
    )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="notebooks/seed_edistance_residual_mechanism.png",
                        help="Output PNG path.")
    parser.add_argument("--svg", default="notebooks/seed_edistance_residual_mechanism.svg",
                        help="Output SVG path. Use an empty string to skip SVG export.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)

    fig = make_figure()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    if args.svg:
        svg = Path(args.svg)
        svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    if args.svg:
        print(f"wrote {args.svg}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script manually**

Run:

```bash
python scripts/plot_seed_edistance_residual_slide.py \
  --out notebooks/seed_edistance_residual_mechanism.png \
  --svg notebooks/seed_edistance_residual_mechanism.svg
```

Expected output:

```text
wrote notebooks/seed_edistance_residual_mechanism.png
wrote notebooks/seed_edistance_residual_mechanism.svg
```

Expected files:
- `notebooks/seed_edistance_residual_mechanism.png`
- `notebooks/seed_edistance_residual_mechanism.svg`

- [ ] **Step 3: Inspect the generated PNG**

Open the PNG visually and confirm:
- Three panels are present.
- The left panel says `N perturbed cells`, not a dataset-specific count.
- The middle panel says `collapsed cloud` and `spread(pred) near zero`.
- The right panel says `fair E-distance`.
- The bottom strip states that E uses residual-restored predictions and PCC-delta/MSE use raw mean predictions.

If any label overlaps or is cut off, adjust only spacing, font size, or `fig.subplots_adjust(...)`.

- [ ] **Step 4: Commit the figure generator**

Run:

```bash
git add scripts/plot_seed_edistance_residual_slide.py \
  notebooks/seed_edistance_residual_mechanism.png \
  notebooks/seed_edistance_residual_mechanism.svg
git commit -m "fig: add seed E-distance residual mechanism slide"
```

## Task 2: Add Tests for Figure Structure and CLI Export

**Files:**
- Create: `tests/test_seed_edistance_residual_slide.py`
- Modify: `scripts/plot_seed_edistance_residual_slide.py` only if tests reveal import/export problems

- [ ] **Step 1: Write the tests**

Use `apply_patch` to create `tests/test_seed_edistance_residual_slide.py` with this complete content:

```python
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from scripts.plot_seed_edistance_residual_slide import make_figure


def _all_text(fig):
    texts = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        texts.extend(t.get_text() for t in ax.texts)
    return "\n".join(texts)


def test_seed_edistance_slide_has_required_labels():
    fig = make_figure()
    txt = _all_text(fig)
    assert len(fig.axes) == 3
    assert "Observed perturbed seeds" in txt
    assert "N perturbed cells" in txt
    assert "Mean-field seed prediction" in txt
    assert "collapsed cloud" in txt
    assert "spread(pred) near zero" in txt
    assert "Prediction + control residual" in txt
    assert "fair E-distance" in txt
    assert "residual-restored predictions for E-distance" in txt
    assert "raw mean predictions for PCC-delta and MSE" in txt


def test_seed_edistance_slide_cli_writes_png_and_svg(tmp_path):
    root = Path(__file__).resolve().parent.parent
    out = tmp_path / "seed_residual.png"
    svg = tmp_path / "seed_residual.svg"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "plot_seed_edistance_residual_slide.py"),
            "--out", str(out),
            "--svg", str(svg),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and out.stat().st_size > 10_000
    assert svg.exists() and svg.stat().st_size > 5_000
    assert "wrote" in result.stdout
```

- [ ] **Step 2: Run the new tests**

Run:

```bash
python -m pytest tests/test_seed_edistance_residual_slide.py -q
```

Expected output:

```text
2 passed
```

If the CLI test fails with `ModuleNotFoundError`, run it from the repository root and verify the test uses `cwd=str(root)`. Do not add package installation requirements for this script.

- [ ] **Step 3: Run the plotting test group**

Run:

```bash
python -m pytest tests/test_seed_edistance_residual_slide.py tests/test_plotting.py -q
```

Expected output:

```text
7 passed
```

If `tests/test_plotting.py::test_demo_script_writes_png` fails with `ModuleNotFoundError: No module named 'spbench'`, that is the existing environment/package issue observed before this plan. Record it in the final handoff and do not change this slide task to mask it.

- [ ] **Step 4: Commit the tests**

Run:

```bash
git add tests/test_seed_edistance_residual_slide.py scripts/plot_seed_edistance_residual_slide.py
git commit -m "test: cover seed E-distance residual slide export"
```

## Task 3: Final Verification and Handoff Notes

**Files:**
- No new files required
- Read: `docs/superpowers/specs/2026-06-25-seed-edistance-residual-slide-design.md`

- [ ] **Step 1: Compare output against the approved spec**

Open:

```bash
sed -n '1,220p' docs/superpowers/specs/2026-06-25-seed-edistance-residual-slide-design.md
```

Confirm the generated figure satisfies all acceptance criteria:
- It uses `N perturbed cells`.
- It distinguishes mean preservation from variance restoration.
- It says residuals come from controls by naming `same-type control residual`.
- It states E uses residual-restored predictions while PCC-delta/MSE use raw mean predictions.

- [ ] **Step 2: Run targeted tests**

Run:

```bash
python -m pytest tests/test_seed_edistance_residual_slide.py -q
```

Expected output:

```text
2 passed
```

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short --branch
```

Expected:
- The branch may be ahead of remote.
- There should be no unstaged changes from the slide implementation.

- [ ] **Step 4: Final response**

Report:
- The PNG path: `notebooks/seed_edistance_residual_mechanism.png`
- The SVG path: `notebooks/seed_edistance_residual_mechanism.svg`
- The targeted test command and result.
- Any known unrelated test failure, especially the pre-existing demo script import issue if it appears.

Use a concise final answer in Chinese.

## Self-Review

Spec coverage:
- One 16:9 technical mechanism slide: Task 1 creates a deterministic three-panel figure.
- `N` instead of `39`: Task 1 labels `N perturbed cells`; Task 2 asserts it.
- Mean preservation vs variance restoration: Task 1 labels `mean unchanged; spread matched`; Task 2 checks the bottom message.
- Control-only residual / no leakage: Task 1 labels the arrow `same-type control residual`; Task 3 checks this against the spec.
- E uses residual-restored predictions while PCC/MSE use raw mean: Task 1 bottom strip and Task 2 assertion cover it.

Incomplete-marker scan:
- No incomplete markers or dangling function references.

Type consistency:
- `make_figure(figsize=(13.4, 7.2))` returns a Matplotlib `Figure`.
- `main(argv=None)` accepts CLI arguments and writes PNG/SVG paths.
- Tests import `make_figure` from the exact script path and invoke the exact CLI file.
