"""Cross-dataset aggregation.

Absolute scores are NOT comparable across datasets — gene count, platform (Visium spot vs
single-cell imaging), and the scoring space each rescale them, so a raw PCC-delta in one space is
not the same currency as in another. So to compare a method ACROSS datasets we (1) normalize each
method's PCC-delta WITHIN each dataset against that dataset's own null level and GT-seed upper
bound, and (2) rank methods per dataset and aggregate the ranks. This avoids aligning
representations or comparing raw scores across datasets.
"""
import numpy as np


def normalized_pcc(pcc, pcc_null, pcc_upper):
    """Map a method's PCC-delta to [0,1] within one dataset/space: 0 = the null level (no-effect,
    PCC-delta ~ 0), 1 = the GT-seed upper bound. This makes PCC comparable across models scored in
    DIFFERENT spaces — each is normalized against its OWN same-space null and upper, so the space's
    gene-weighting cancels.

        norm = clip((pcc - pcc_null) / (pcc_upper - pcc_null), 0, 1)

    Returns nan when there is no positive headroom to normalize against — non-finite gap, or
    pcc_upper <= pcc_null (the GT-seed upper is not above the null, so the normalization would be
    meaningless / sign-flipped)."""
    gap = pcc_upper - pcc_null
    if not np.isfinite(gap) or gap < 1e-12:
        return float("nan")
    return float(np.clip((pcc - pcc_null) / gap, 0.0, 1.0))


def cross_dataset_rank(per_dataset):
    """Rank methods within each dataset and aggregate across datasets.

    per_dataset : {dataset: {method: {"pcc": float, "null": float, "upper": float}}}
                  ("upper" is the per-dataset GT-seed ceiling; "null" the no-effect level ~0).
    Returns {
      "normalized": {dataset: {method: norm in [0,1] or nan}},
      "ranks":      {dataset: {method: rank}}  (rank 1 = highest pcc = best within that dataset),
      "aggregate":  {method: {"mean_rank", "mean_norm", "n_datasets"}}  (lower mean_rank = better),
    }
    """
    normalized, ranks, acc = {}, {}, {}
    for ds, methods in per_dataset.items():
        norm = {m: normalized_pcc(s["pcc"], s.get("null", 0.0), s.get("upper", float("nan")))
                for m, s in methods.items()}
        normalized[ds] = norm
        ordered = sorted(methods, key=lambda m: methods[m]["pcc"], reverse=True)  # descending pcc: best first
        ranks[ds] = {m: i + 1 for i, m in enumerate(ordered)}
        for m in methods:
            a = acc.setdefault(m, {"ranks": [], "norms": []})
            a["ranks"].append(ranks[ds][m])
            if np.isfinite(norm[m]):
                a["norms"].append(norm[m])
    aggregate = {
        m: {"mean_rank": float(np.mean(a["ranks"])),
            "mean_norm": float(np.mean(a["norms"])) if a["norms"] else float("nan"),
            "n_datasets": len(a["ranks"])}
        for m, a in acc.items()
    }
    return {"normalized": normalized, "ranks": ranks, "aggregate": aggregate}


def rank_from_results(results_by_dataset, methods=("model+base", "model+learned"),
                      null_key="null", upper_key="GT+learned"):
    """Build the per_dataset dict from {dataset: run_benchmark res} by averaging each method's
    pcc / null / upper over that dataset's perturbations (res['compare']), then cross_dataset_rank it.
    `upper_key` is the GT-seed cell whose niche PCC-delta is the per-dataset upper bound."""
    per = {}
    for ds, res in results_by_dataset.items():
        cmp = res.get("compare", {})
        md = {}
        for m in methods:
            ps = [c["pcc"][m] for c in cmp.values()
                  if m in c.get("pcc", {}) and np.isfinite(c["pcc"][m])]
            if not ps:
                continue
            nulls = [c["pcc"][null_key] for c in cmp.values()
                     if null_key in c.get("pcc", {}) and np.isfinite(c["pcc"][null_key])]
            ups = [c["pcc"][upper_key] for c in cmp.values()
                   if upper_key in c.get("pcc", {}) and np.isfinite(c["pcc"][upper_key])]
            md[m] = {"pcc": float(np.mean(ps)),
                     "null": float(np.mean(nulls)) if nulls else 0.0,
                     "upper": float(np.mean(ups)) if ups else float("nan")}
        per[ds] = md
    return cross_dataset_rank(per)
