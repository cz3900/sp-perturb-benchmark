import numpy as np
import matplotlib.pyplot as plt

# colour convention used across the result figures
_SIG = "#4C78A8"     # significant perturbations (blue)
_NON = "#BAB0AC"     # non-significant perturbations (gray)

def _lv(res):
    """{perturbation: learned_value} from a run_benchmark result."""
    return {p: res["attribution"][p]["learned_value"] for p in res["attribution"]}

def plot_2x2(grid, title=""):
    """Heatmap of the 2x2 E-distance grid."""
    M = np.array([[grid["1"]["energy_prop"], grid["2"]["energy_prop"]],
                  [grid["3"]["energy_prop"], grid["4"]["energy_prop"]]])
    fig, ax = plt.subplots(figsize=(4, 3.2))
    im = ax.imshow(M, cmap="viridis_r")
    ax.set_xticks([0, 1], ["baseline prop", "learned prop"])
    ax.set_yticks([0, 1], ["GT seed", "model seed"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w")
    ax.set_title(f"2x2 E-distance {title}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig

def plot_attribution(attrib: dict):
    """Bar chart: per-perturbation end-to-end score + seed_cost + learned_value."""
    perts = list(attrib)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(perts))
    ax.bar(x - 0.25, [attrib[p]["end_to_end"] for p in perts], 0.25, label="end-to-end")
    ax.bar(x, [attrib[p]["seed_cost"] for p in perts], 0.25, label="seed cost")
    ax.bar(x + 0.25, [attrib[p]["learned_value"] for p in perts], 0.25, label="learned value")
    ax.set_xticks(x, perts)
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(); ax.set_title("2x2 attribution per perturbation")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Result figures that show the DIFFERENCES (not absolute E-distance), aggregated
# across perturbations, contrasting significant vs non-significant. These answer
# "does learned propagation help, and only where there is real spatial signal?"
# ---------------------------------------------------------------------------

def _legend_sig(ax):
    ax.scatter([], [], color=_SIG, label="significant")
    ax.scatter([], [], color=_NON, label="non-significant")
    ax.legend(loc="best", fontsize=9)

def plot_learned_value(res, significant=None, title="Learned propagation value (e1 - e2)"):
    """Figure A — horizontal bars of learned_value for every perturbation, sorted,
    coloured by significance. >0 means the learned GCN beats the Gaussian baseline.
    Outliers and the significant/non-significant split are visible at a glance."""
    sig = set(significant or [])
    lv = _lv(res)
    order = sorted(lv, key=lambda p: lv[p])
    colors = [_SIG if p in sig else _NON for p in order]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(order))))
    ax.barh(range(len(order)), [lv[p] for p in order], color=colors)
    ax.set_yticks(range(len(order)), order)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_xlabel("learned_value = e1 - e2   (>0: GCN better than Gaussian)")
    ax.set_title(title)
    if sig:
        _legend_sig(ax)
    fig.tight_layout()
    return fig

def plot_significance_contrast(res, significant, title="Learned propagation helps where there is signal"):
    """Figure B — learned_value distribution for significant vs non-significant groups
    (box + jittered points), annotated with a sign-test p-value on the significant group.
    This is the headline 'does it work' figure: the contrast is the proof."""
    from scipy.stats import binomtest
    sig = set(significant)
    lv = _lv(res)
    grp_sig = [lv[p] for p in lv if p in sig]
    grp_non = [lv[p] for p in lv if p not in sig]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.boxplot([grp_non, grp_sig], positions=[0, 1], widths=0.5, showfliers=False)
    rng = np.random.default_rng(0)
    for i, (d, col) in enumerate([(grp_non, _NON), (grp_sig, _SIG)]):
        if d:
            ax.scatter(rng.normal(i, 0.05, size=len(d)), d, color=col, alpha=0.8, zorder=3)
    ax.set_xticks([0, 1], [f"non-significant\n(n={len(grp_non)})", f"significant\n(n={len(grp_sig)})"])
    ax.axhline(0, color="k", lw=0.9)
    ax.set_ylabel("learned_value (e1 - e2)")
    sub = ""
    if grp_sig:
        pos = sum(v > 0 for v in grp_sig)
        p = binomtest(pos, len(grp_sig), 0.5, alternative="greater").pvalue
        sub = f"\nsignificant group: {pos}/{len(grp_sig)} positive, sign-test p = {p:.3f}"
    ax.set_title(title + sub, fontsize=11)
    fig.tight_layout()
    return fig

def plot_slope(res, significant=None, title="Baseline -> learned, per perturbation"):
    """Figure C — slope plot from e1 (baseline prop) to e2 (learned prop) for each
    perturbation. A downward slope means the GCN is better; consistency of direction and
    any outlier (e.g. a line sitting far above the rest) are immediately visible."""
    sig = set(significant or [])
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    for p in res["grids"]:
        e1 = res["grids"][p]["1"]["energy_prop"]
        e2 = res["grids"][p]["2"]["energy_prop"]
        col = _SIG if p in sig else _NON
        ax.plot([0, 1], [e1, e2], "-o", color=col, alpha=0.75, markersize=4)
        ax.annotate(p, (1.02, e2), fontsize=7, va="center", color=col)
    ax.set_xlim(-0.15, 1.4)
    ax.set_xticks([0, 1], ["baseline (e1)", "learned (e2)"])
    ax.set_ylabel("E-distance (lower = better)")
    ax.set_title(title + "\n(downward = GCN better)", fontsize=11)
    if sig:
        _legend_sig(ax)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Baseline-gain figures. gain = e_null - e_method (>0 beats predicting 'no effect').
# These replace the old skill score: same currency (energy distance), no ratios.
# Need res['compare'] from run_benchmark(..., compare=True).
# ---------------------------------------------------------------------------

# methods plotted left->right: most deployable first, oracle handled separately as a ceiling line
_GAIN_METHODS = ["model+learned", "model+base", "GT+learned", "GT+base"]
_GAIN_COLORS = {"model+learned": _SIG, "model+base": "#72B7B2",
                "GT+learned": "#B0A8D0", "GT+base": _NON}

def plot_baseline_gain(res, title="Each method vs the no-effect baseline (all perturbations)"):
    """HEADLINE — one box (+ jittered points, one per perturbation) per method, of
    gain = e_null - e_method. The horizontal line at 0 IS the no-effect baseline: a method is
    useful only where its points sit ABOVE 0. The dashed line is the oracle ceiling (the best a
    non-leaking model could reach). Answers 'does each method beat doing nothing, and by how much,
    across all genes?'."""
    cmp = res.get("compare", {})
    perts = [p for p in cmp if cmp[p].get("gain")]
    data = [[cmp[p]["gain"].get(m, float("nan")) for p in perts] for m in _GAIN_METHODS]
    data = [[v for v in col if v == v] for col in data]           # drop NaNs
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.boxplot(data, positions=range(len(_GAIN_METHODS)), widths=0.5, showfliers=False)
    rng = np.random.default_rng(0)
    for i, (m, col) in enumerate(zip(_GAIN_METHODS, [_GAIN_COLORS[m] for m in _GAIN_METHODS])):
        d = data[i]
        if d:
            ax.scatter(rng.normal(i, 0.06, size=len(d)), d, color=col, alpha=0.85, zorder=3, s=22)
    ax.axhline(0, color="k", lw=1.1)
    ax.text(len(_GAIN_METHODS) - 0.5, 0, " baseline (no effect)", va="bottom", ha="right", fontsize=8)
    oracle = [cmp[p]["gain"]["oracle"] for p in perts if "oracle" in cmp[p].get("gain", {})]
    if oracle:
        m = float(np.nanmean(oracle))
        ax.axhline(m, color="0.4", lw=1.0, ls="--")
        ax.text(0, m, " ceiling (best possible)", va="bottom", ha="left", fontsize=8, color="0.4")
    ax.set_xticks(range(len(_GAIN_METHODS)), _GAIN_METHODS, rotation=12)
    ax.set_ylabel("gain = e_null − e   (>0 beats 'no effect')")
    ax.set_title(f"{title}\n{len(perts)} perturbations", fontsize=11)
    fig.tight_layout()
    return fig

def plot_aggregate_2x2(res, title="Mean gain over the no-effect baseline (all perturbations)"):
    """One summary 2x2: each cell is the per-gene gain (= e_null - e) averaged across all
    perturbations, so it is normalised to each gene's own baseline (0 = no better than 'no
    effect'; >0 beats it). The COLUMN difference is the mean learned_value (learned vs Gaussian
    propagation), the ROW difference is the mean seed_cost — baseline anchoring and the 2x2
    internal contrasts in one grid. Cells also show how many of N perturbations individually beat
    the baseline (gain>0). A companion to plot_baseline_gain, which shows the full spread."""
    cmp = res.get("compare", {})
    perts = [p for p in cmp if cmp[p].get("gain")]
    cells = {"1": "GT+base", "2": "GT+learned", "3": "model+base", "4": "model+learned"}
    def col(k):
        return [cmp[p]["gain"][cells[k]] for p in perts if cells[k] in cmp[p]["gain"]]
    M = np.array([[np.nanmean(col("1")), np.nanmean(col("2"))],
                  [np.nanmean(col("3")), np.nanmean(col("4"))]])
    nbeat = {k: sum(v > 0 for v in col(k)) for k in cells}
    N = len(perts)
    vmax = float(np.nanmax(np.abs(M))) or 1.0
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax)   # red<0 (loses to baseline), green>0
    ax.set_xticks([0, 1], ["baseline prop", "learned prop"])
    ax.set_yticks([0, 1], ["GT seed", "model seed"])
    order = [["1", "2"], ["3", "4"]]
    for i in range(2):
        for j in range(2):
            k = order[i][j]
            ax.text(j, i, f"{M[i, j]:+.3f}\n{nbeat[k]}/{N} beat", ha="center", va="center",
                    fontsize=10, fontweight="bold")
    ax.set_title(f"{title}\n>0 (green) beats baseline; col diff = learned value, row diff = seed cost",
                 fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean gain (e_null − e)")
    fig.tight_layout()
    return fig

def plot_gain_per_perturbation(res, significant=None, method="model+learned",
                               title="Deployable model vs baseline, per perturbation"):
    """DETAIL — horizontal bars of gain = e_null - e for the deployable method, one per
    perturbation, sorted, coloured by significance. >0 (right of the line) = this gene's niche is
    predicted better than assuming 'no effect'. Shows WHICH genes (if any) the pipeline wins on."""
    cmp = res.get("compare", {})
    sig = set(significant or [])
    g = {p: cmp[p]["gain"].get(method) for p in cmp if cmp[p].get("gain")}
    g = {p: v for p, v in g.items() if v == v}
    order = sorted(g, key=lambda p: g[p])
    colors = [_SIG if p in sig else _NON for p in order]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(order))))
    ax.barh(range(len(order)), [g[p] for p in order], color=colors)
    ax.set_yticks(range(len(order)), order)
    ax.axvline(0, color="k", lw=1.1)
    ax.set_xlabel(f"gain = e_null − e[{method}]   (>0: beats 'no effect')")
    ax.set_title(title)
    if sig:
        _legend_sig(ax)
    fig.tight_layout()
    return fig

def plot_seed_vs_learned(res, significant=None, title="Attribution: seed cost vs learned value"):
    """Figure D (optional) — scatter of seed_cost (x) vs learned_value (y), one point per
    perturbation, coloured by significance. Shows the error is mostly on the propagation
    side (points near x=0) and that learned beats baseline (points with y>0)."""
    sig = set(significant or [])
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for p in res["attribution"]:
        a = res["attribution"][p]
        col = _SIG if p in sig else _NON
        ax.scatter(a["seed_cost"], a["learned_value"], color=col, zorder=3)
        ax.annotate(p, (a["seed_cost"], a["learned_value"]), fontsize=7, color=col)
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("seed_cost = e3 - e1   (seed prediction penalty)")
    ax.set_ylabel("learned_value = e1 - e2   (learned propagation gain)")
    ax.set_title(title, fontsize=11)
    if sig:
        _legend_sig(ax)
    fig.tight_layout()
    return fig
