"""Turn run_benchmark output into per-dataset seed/niche method-comparison box plots.

Each method's box = its pooled per-repeat matched-n energy distances (from `e_samples`).
GCN is the learned-prop method (`model+learned`); shown named "GCN" in the niche plot.
Dashed lines = no-effect null floor (gray) and the per-prop GT-seed upper bound (teal)."""
import numpy as np

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


def collect_delta(res, kind):
    """PCC-delta per method for one task. kind='seed' -> res['seed'][p]['pcc_delta'] (box 'seed
    model'); kind='niche' -> res['compare'][p]['pcc'][method] for the deployable cells + externals.
    Returns ({label: [pcc per guide]}, {'perfect':1.0,'no-corr':0.0})."""
    boxes = {}
    if kind == "seed":
        vals = [s["pcc_delta"] for s in res.get("seed", {}).values()
                if np.isfinite(s.get("pcc_delta", np.nan))]
        if vals:
            boxes["seed model"] = vals
    else:
        cmp = res.get("compare", {})
        for m, label in (("model+base", "Gaussian"), ("model+learned", "GCN")):
            vals = [c["pcc"][m] for c in cmp.values() if np.isfinite(c.get("pcc", {}).get(m, np.nan))]
            if vals:
                boxes[label] = vals
        for nm in external_methods(res):
            vals = [c["pcc"][nm] for c in cmp.values() if np.isfinite(c.get("pcc", {}).get(nm, np.nan))]
            if vals:
                boxes[nm] = vals
    return boxes, {"perfect": 1.0, "no-corr": 0.0}


def plot_delta(res, figsize=(11, 4.3)):
    """Primary view: PCC-delta (mean-shift direction; higher=better, bounded). seed | niche, same
    axes. Dashed: perfect=1, no-corr=0."""
    import matplotlib.pyplot as plt
    sb, _ = collect_delta(res, "seed")
    nb, _ = collect_delta(res, "niche")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, boxes, title in ((ax0, sb, "seed — PCC-delta"), (ax1, nb, "niche — PCC-delta")):
        labels = list(boxes)
        if labels:
            ax.boxplot([np.asarray(boxes[l], float) for l in labels], tick_labels=labels, showfliers=False)
        ax.axhline(1.0, ls="--", lw=1.2, color="#1d9e75", label="perfect")
        ax.axhline(0.0, ls="--", lw=1.2, color="#888888", label="no-corr")
        ax.set_ylabel("PCC-delta (higher = better)"); ax.set_title(title)
        ax.legend(fontsize=8, loc="best"); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


# niche cell key -> column label for the PCC-delta summary (the locked scoring currency).
_NICHE_COLS = {"model+base": "Gaussian", "model+learned": "GCN"}


def summary_table(res):
    """Per-guide PCC-delta summary — the single locked scoring currency (Task 0).
    One row per guide: seed PCC-delta + niche PCC-delta for the deployable cells
    (Gaussian/GCN) and every external/end-to-end model. Returns list[dict].

    External method names are derived from the `pcc` dict keys (not e_samples), so the table works
    whether or not box-plot samples are present."""
    seed = res.get("seed", {})
    cmp = res.get("compare", {})
    guides = sorted(set(seed) | set(cmp))
    ext = sorted({k for c in cmp.values() for k in c.get("pcc", {})} - _STD_METHODS)
    rows = []
    for g in guides:
        row = {"guide": g, "seed_pcc_delta": seed.get(g, {}).get("pcc_delta", float("nan"))}
        pcc = cmp.get(g, {}).get("pcc", {})
        for cell, label in _NICHE_COLS.items():
            row[f"niche_{label}"] = pcc.get(cell, float("nan"))
        for nm in ext:
            row[f"niche_{nm}"] = pcc.get(nm, float("nan"))
        rows.append(row)
    return rows
