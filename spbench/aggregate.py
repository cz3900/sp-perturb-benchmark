"""Cross-dataset aggregation.

Absolute E-distance is NOT comparable across datasets — gene count, platform (Visium spot vs
single-cell imaging), and neighbourhood aggregation each rescale it by orders of magnitude (a
Visium whole-transcriptome niche E ~ hundreds, a MERFISH aggregated niche E < 1). So to compare a
method ACROSS datasets we (1) normalize each method's score WITHIN each dataset against that
dataset's own null floor and oracle ceiling, and (2) rank methods per dataset and aggregate the
ranks. This avoids aligning representations or comparing raw E across datasets.
"""
import numpy as np


def normalized_score(e, e_null, e_oracle):
    """Map a method's energy distance to [0, 1] within one dataset: 0 = the null floor (no-effect
    baseline), 1 = the oracle / GT-seed ceiling (best non-leaking). Higher = better.

        norm = clip((e_null - e) / (e_null - e_oracle), 0, 1)

    Returns nan when the floor/ceiling gap is degenerate (non-finite or ~0)."""
    gap = e_null - e_oracle
    if not np.isfinite(gap) or abs(gap) < 1e-12:
        return float("nan")
    return float(np.clip((e_null - e) / gap, 0.0, 1.0))


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

    per_dataset : {dataset: {method: {"e": float, "null": float, "oracle": float}}}
                  ("oracle" may be any per-dataset ceiling, e.g. the GT-seed upper bound).
    Returns {
      "normalized": {dataset: {method: norm in [0,1] or nan}},
      "ranks":      {dataset: {method: rank}}  (rank 1 = lowest e = best within that dataset),
      "aggregate":  {method: {"mean_rank", "mean_norm", "n_datasets"}}  (lower mean_rank = better),
    }
    """
    normalized, ranks, acc = {}, {}, {}
    for ds, methods in per_dataset.items():
        norm = {m: normalized_score(s["e"], s.get("null", float("nan")), s.get("oracle", float("nan")))
                for m, s in methods.items()}
        normalized[ds] = norm
        ordered = sorted(methods, key=lambda m: methods[m]["e"])      # ascending e: best first
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
                      null_key="null", oracle_key="oracle"):
    """Build the per_dataset dict from {dataset: run_benchmark res} by averaging each method's e /
    null / oracle over that dataset's perturbations (res['compare']), then cross_dataset_rank it."""
    per = {}
    for ds, res in results_by_dataset.items():
        cmp = res.get("compare", {})
        md = {}
        for m in methods:
            es = [c["e"][m] for c in cmp.values() if m in c.get("e", {})]
            if not es:
                continue
            nulls = [c["e"][null_key] for c in cmp.values() if null_key in c.get("e", {})]
            orcs = [c["e"][oracle_key] for c in cmp.values() if oracle_key in c.get("e", {})]
            md[m] = {"e": float(np.mean(es)),
                     "null": float(np.mean(nulls)) if nulls else float("nan"),
                     "oracle": float(np.mean(orcs)) if orcs else float("nan")}
        per[ds] = md
    return cross_dataset_rank(per)
