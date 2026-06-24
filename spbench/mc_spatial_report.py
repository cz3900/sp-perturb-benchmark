"""Stratified scoring report: quadrant x dimension, with the Inert quadrant as negative control.

Consumes records that already carry an MC-spatial `quadrant` label per perturbation (produced by
spbench.mc_spatial_join.join_quadrants) plus per-dimension gain fields (gain = e_null - e_method,
the headline quantity in spbench.compare; >0 beats the no-effect baseline).

Stratification rule:
  D1 (self/seed)  is meaningful only where there is X-signal  -> {X-Only, Both}
  D2 (niche)      is meaningful only where there is Y-signal  -> {Y-Only, Both}
  Inert           is the NEGATIVE CONTROL for every dimension -> a good model predicts ~no
                  effect there (mean gain ~ 0); positive gain in Inert = hallucinated effect.

This avoids the dilution problem (do not average niche gain over perturbations with no niche
signal) and gives a rigorous 2D negative control instead of randomly sampled non-significant
guides. Stays numpy-native (no pandas) to match the rest of spbench.
"""
import numpy as np

from .mc_spatial_join import BOTH, X_ONLY, Y_ONLY, INERT

# Which quadrants count toward each dimension's signal group.
DIMENSION_QUADRANTS = {
    "d1": {X_ONLY, BOTH},
    "d2": {Y_ONLY, BOTH},
}

REPORT_FIELDS = (
    "dimension",
    "quadrant_group",
    "n",
    "mean_gain",
    "frac_beat",
    "is_negative_control",
)


def _summarize(gains):
    """(n, mean_gain, frac_beat) for a sequence of gains; NaN mean/frac on an empty group."""
    g = np.asarray(list(gains), dtype=float)
    n = int(g.size)
    if n == 0:
        return 0, float("nan"), float("nan")
    return n, float(np.mean(g)), float(np.mean(g > 0))


def stratified_report(records, dim_gain_fields=None):
    """Build the quadrant x dimension stratified report (long form: list of dict rows).

    `records`: iterable of dicts each with a 'quadrant' label and the per-dimension gain fields.
    `dim_gain_fields`: {dimension: gain-field-name}, default {"d1": "gain_d1", "d2": "gain_d2"}.

    For each dimension: one 'signal' row aggregating gain over that dimension's signal quadrants,
    and one negative-control row aggregating the same gain over the Inert quadrant. Each row's
    keys are exactly REPORT_FIELDS, in order.
    """
    if dim_gain_fields is None:
        dim_gain_fields = {"d1": "gain_d1", "d2": "gain_d2"}
    records = list(records)
    if records and not all("quadrant" in r for r in records):
        raise ValueError("every record must carry a 'quadrant' field (run join_quadrants first)")

    rows = []
    for dim, gain_field in dim_gain_fields.items():
        if records and not all(gain_field in r for r in records):
            raise ValueError(f"missing gain field {gain_field!r} for dimension {dim!r}")
        sig_quadrants = DIMENSION_QUADRANTS[dim]

        sig_gains = [r[gain_field] for r in records if r["quadrant"] in sig_quadrants]
        n, mean_gain, frac_beat = _summarize(sig_gains)
        rows.append({
            "dimension": dim,
            "quadrant_group": "+".join(sorted(sig_quadrants)),
            "n": n,
            "mean_gain": mean_gain,
            "frac_beat": frac_beat,
            "is_negative_control": False,
        })

        inert_gains = [r[gain_field] for r in records if r["quadrant"] == INERT]
        n_i, mean_i, frac_i = _summarize(inert_gains)
        rows.append({
            "dimension": dim,
            "quadrant_group": "Inert",
            "n": n_i,
            "mean_gain": mean_i,
            "frac_beat": frac_i,
            "is_negative_control": True,
        })

    return rows
