"""Turn run_benchmark output into per-dataset seed/niche method-comparison box plots.

Each method's box = its pooled per-repeat matched-n energy distances (from `e_samples`).
GCN is the learned-prop method (`model+learned`); shown named "GCN" in the niche plot.
Dashed lines = no-effect null and oracle ceiling (mean energy)."""
import numpy as np

# 2x2 method key -> niche-plot label. GCN = learned prop (model seed + GCN); Gaussian = baseline prop.
PROP_LABELS = {"model+base": "Gaussian", "model+learned": "GCN",
               "GT+base": "Gaussian (GT seed)", "GT+learned": "GCN (GT seed)"}
# dashed reference lines: no-effect null (gray) and oracle ceiling (teal)
_DASH_COLORS = {"null": "#888888", "oracle": "#1d9e75"}


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
    """{label: pooled per-repeat energy} for the model seed, plus {'null': mean} dashed."""
    boxes, dashed = {}, {}
    seed = res.get("seed", {})
    pooled = [s for v in seed.values() for s in v.get("e_samples", {}).get("model", [])]
    if pooled:
        boxes[model_label] = np.asarray(pooled, float)
    null_p = [s for v in seed.values() for s in v.get("e_samples", {}).get("null", [])]
    if null_p:
        dashed["null"] = float(np.nanmean(null_p))
    return boxes, dashed


def _draw_boxes(ax, boxes, dashed, ylabel, title):
    labels = list(boxes)
    data = [np.log(np.clip(boxes[l], 1e-9, None)) for l in labels]
    if data:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    for name, val in dashed.items():
        ax.axhline(np.log(max(val, 1e-9)), ls="--", lw=1.2,
                   color=_DASH_COLORS.get(name, "#888888"), label=name)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if dashed:
        ax.legend(fontsize=8, loc="best")
    ax.tick_params(axis="x", rotation=30)


def plot_seed_prop(res, figsize=(11, 4.2)):
    """One figure, two box plots: Δseed (left) and Δniche (right). x = methods (GCN named),
    box = pooled per-repeat log energy distance, dashed = null / oracle baselines."""
    import matplotlib.pyplot as plt
    sb, sd = collect_seed_samples(res)
    pb, pd = collect_prop_samples(res)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    _draw_boxes(ax1, sb, sd, "log E-distance", "seed (D1)")
    _draw_boxes(ax2, pb, pd, "log E-distance", "niche (D2)")
    fig.tight_layout()
    return fig
