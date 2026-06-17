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

def plot_skill_leaderboard(res, title="Propagation skill — fraction of recoverable signal captured"):
    """Headline figure: per-perturbation 0..100% skill for the two deployable models (Gaussian
    baseline vs learned GCN), shown only for perturbations with real signal. skill 0 = no better
    than predicting 'no effect' (the control niche); 100 = perfect (at the noise floor).
    Needs res['skill'] from run_benchmark(..., compute_skill=True)."""
    skill = res.get("skill", {})
    items = [(p, skill[p]) for p in skill if skill[p].get("has_signal")]
    dropped = [p for p in skill if not skill[p].get("has_signal")]
    items.sort(key=lambda t: (t[1].get("learned") if t[1].get("learned") == t[1].get("learned") else -1))
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(items) + 1)))
    clip = lambda v: float("nan") if v != v else max(-100.0, min(100.0, 100 * v))
    if items:
        names = [p for p, _ in items]
        y = np.arange(len(names))
        base = [clip(items[i][1]["baseline"]) for i in range(len(items))]
        lrn = [clip(items[i][1]["learned"]) for i in range(len(items))]
        ax.barh(y - 0.2, base, 0.4, color=_NON, label="Gaussian baseline")
        ax.barh(y + 0.2, lrn, 0.4, color=_SIG, label="learned GCN")
        ax.set_yticks(y, names)
        ax.set_xlim(-105, 105)
        ax.legend(loc="lower right", fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("skill (%)  [clipped to ±100]   <0 = worse than 'no effect',  100 = perfect")
    note = (f"   ({len(dropped)} of {len(skill)} dropped: no reliable signal)") if dropped else ""
    ax.set_title(f"{title}\n{len(items)} perturbations with signal{note}", fontsize=11)
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
