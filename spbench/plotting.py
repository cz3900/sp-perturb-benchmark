"""Turn run_benchmark output into per-dataset seed/niche method-comparison box plots.

The single scoring currency is PCC-delta (mean-shift direction; higher = better, bounded). Each
method's box = its per-guide PCC-delta. Dashed reference lines: perfect = 1, no-corr = 0.
"""
import numpy as np

# standard scoring keys in res['compare'][p]['pcc']: the 2x2 cells + null. Anything else is an
# external / end-to-end model fed via run_benchmark(external_models=...) / extra=.
_STD_METHODS = {"GT+base", "GT+learned", "model+base", "model+learned", "null"}


def external_methods(res):
    """Names of external/end-to-end methods present in res['compare'] (anything in the pcc dict
    that is not a standard 2x2 cell / null)."""
    cmp = res.get("compare", {})
    names = set()
    for c in cmp.values():
        names |= (set(c.get("pcc", {})) - _STD_METHODS)
    return sorted(names)


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

    External method names are derived from the `pcc` dict keys, so the table works whether or not
    box-plot samples are present."""
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
