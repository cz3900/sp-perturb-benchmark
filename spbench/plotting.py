"""Turn run_benchmark output into per-dataset seed/niche method-comparison box plots.

Each method's box = its pooled per-repeat matched-n energy distances (from `e_samples`).
GCN is the learned-prop method (`model+learned`); shown named "GCN" in the niche plot.
Dashed lines = no-effect null floor (gray) and the per-prop GT-seed upper bound (teal)."""
import numpy as np

# 2x2 method key -> niche-plot label. GCN = learned prop (model seed + GCN); Gaussian = baseline prop.
PROP_LABELS = {"model+base": "Gaussian", "model+learned": "GCN",
               "GT+base": "Gaussian (GT seed)", "GT+learned": "GCN (GT seed)"}
# dashed reference lines: no-effect null floor (gray) and GT-seed upper bound (teal)
_DASH_COLORS = {"null (floor)": "#888888", "GT-seed (upper)": "#1d9e75"}

# niche board per prop tier: (box label, box 2x2 cell, upper-bound 2x2 cell = GT seed + that prop)
NICHE_TIERS = {
    "base":    ("Gaussian (seed-only)",   "model+base",    "GT+base"),     # tier 1: seed-only + Gaussian, upper = cell-1
    "learned": ("GCN (end-to-end mock)",  "model+learned", "GT+learned"),  # tier 2: end-to-end mock, upper = cell-2
}

# standard scoring keys in res['compare'][p]['e_samples']: the 2x2 cells + null + oracle. Anything
# else is an external / end-to-end model fed via run_benchmark(external_models=...) / extra=.
_STD_METHODS = {"GT+base", "GT+learned", "model+base", "model+learned", "null", "oracle"}


def external_methods(res):
    """Names of external/end-to-end methods present in res['compare'] (anything in e_samples that
    is not a standard 2x2 cell / null / oracle)."""
    cmp = res.get("compare", {})
    names = set()
    for c in cmp.values():
        names |= (set(c.get("e_samples", {})) - _STD_METHODS)
    return sorted(names)


def collect_niche_tier(res, tier):
    """One niche board: box = pooled per-repeat energy of the tier's model cell; dashed =
    null (floor) and GT-seed (per-prop upper bound = GT seed + that prop). For the 'learned'
    (end-to-end) tier, also add one box per external/end-to-end method (keyed by its name)."""
    label, box_cell, upper_cell = NICHE_TIERS[tier]
    cmp = res.get("compare", {})
    pooled = [s for c in cmp.values() for s in c.get("e_samples", {}).get(box_cell, [])]
    boxes = {label: np.asarray(pooled, float)} if pooled else {}
    if tier == "learned":                                  # external/end-to-end models on this board
        for nm in external_methods(res):
            ext = [s for c in cmp.values() for s in c.get("e_samples", {}).get(nm, [])]
            if ext:
                boxes[nm] = np.asarray(ext, float)
    dashed = {}
    nulls = [c["e"]["null"] for c in cmp.values() if "null" in c.get("e", {})]
    uppers = [c["e"][upper_cell] for c in cmp.values() if upper_cell in c.get("e", {})]
    if nulls:
        dashed["null (floor)"] = float(np.nanmean(nulls))
    if uppers:
        dashed["GT-seed (upper)"] = float(np.nanmean(uppers))
    return boxes, dashed


def collect_prop_samples(res, box_methods=("model+base", "model+learned"),
                         dashed_methods=("null", "oracle")):
    """{label: pooled per-repeat energy array} for the box methods, plus {name: mean energy}
    for the dashed baselines. Pools across all perturbations in res['compare']."""
    boxes, dashed = {}, {}
    cmp = res.get("compare", {})
    for m in box_methods:
        pooled = []
        for c in cmp.values():
            pooled += list(c.get("e_samples", {}).get(m, []))
        if pooled:
            boxes[PROP_LABELS.get(m, m)] = np.asarray(pooled, float)
    for m in dashed_methods:
        vals = [c["e"][m] for c in cmp.values() if m in c.get("e", {})]
        if vals:
            dashed[m] = float(np.nanmean(vals))
    return boxes, dashed


def collect_seed_samples(res, model_label="seed model"):
    """{label: pooled per-repeat energy} for the model seed, plus {'null (floor)': mean}
    dashed (seed board has a floor only, no upper bound: GT seed = observed -> energy 0)."""
    boxes, dashed = {}, {}
    seed = res.get("seed", {})
    pooled = [s for v in seed.values() for s in v.get("e_samples", {}).get("model", [])]
    if pooled:
        boxes[model_label] = np.asarray(pooled, float)
    null_p = [s for v in seed.values() for s in v.get("e_samples", {}).get("null", [])]
    if null_p:
        dashed["null (floor)"] = float(np.nanmean(null_p))
    return boxes, dashed


def _draw_boxes(ax, boxes, dashed, ylabel, title):
    labels = list(boxes)
    # linear E-distance (energy >= 0, no log): niche E < 1 would be negative in log space,
    # which is confusing; energy is always non-negative so plot it directly.
    data = [np.asarray(boxes[l], float) for l in labels]
    if data:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    for name, val in dashed.items():
        ax.axhline(val, ls="--", lw=1.2,
                   color=_DASH_COLORS.get(name, "#888888"), label=name)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if dashed:
        ax.legend(fontsize=8, loc="best")
    ax.tick_params(axis="x", rotation=30)


def plot_seed_prop(res, figsize=(15, 4.3)):
    """Three boards: seed (all models, floor only), niche seed-only+Gaussian (upper=cell-1),
    niche end-to-end/GCN-mock (upper=cell-2). E-distance (linear, >=0), lower=better."""
    import matplotlib.pyplot as plt
    sb, sd = collect_seed_samples(res)
    b1, d1 = collect_niche_tier(res, "base")
    b2, d2 = collect_niche_tier(res, "learned")
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)
    _draw_boxes(ax0, sb, sd, "E-distance", "seed — all models (floor = null)")
    _draw_boxes(ax1, b1, d1, "E-distance", "niche · seed-only + Gaussian")
    _draw_boxes(ax2, b2, d2, "E-distance", "niche · end-to-end (GCN mock)")
    fig.tight_layout()
    return fig
