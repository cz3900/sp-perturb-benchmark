"""Join MC-spatial perturbation-significance (four-quadrant) labels onto benchmark scoring records.

MC-spatial (/Users/cz/Documents/ZengLab/MC-spatial, package `mc_spatial`) runs a DESCRIPTIVE
significance test per perturbation on two axes vs a permutation null and writes
`*_Dual_Metrics.csv` (cols: guide, z_score_global, p_val_global, p_adj_global,
z_score_specificity, p_val_specificity, p_adj_specificity, n_perturbed):

    X = Self Specificity  -> p_val_specificity   (a real D1 self-expression effect exists)
    Y = Niche Impact      -> p_val_global        (a real D2 niche-expression effect exists)

This module is the OFFLINE-CSV handoff: we read that CSV with the stdlib `csv` reader (the
benchmark depends on numpy/scipy/h5py, NOT pandas -- see pyproject.toml; every spbench module is
numpy-native) and tag each perturbation with its quadrant, so the scorer can report stratified by
quadrant x dimension and use the Inert quadrant as a rigorous negative control.

Quadrant logic is a standalone re-implementation of mc_spatial/visualize.py::classify_guide
(sig = p < p_cutoff, STRICT; same four label strings) so this package does not import mc_spatial
(which is not installed in this env).
"""
import csv

# Label strings MUST match mc_spatial/visualize.py::classify_guide exactly.
BOTH = "Both (Systemic)"
X_ONLY = "X-Only (Self)"
Y_ONLY = "Y-Only (Niche)"
INERT = "Inert"
UNKNOWN = "Unknown"

# p_mode -> (specificity column, global column) in *_Dual_Metrics.csv.
_P_COLS = {
    "raw": ("p_val_specificity", "p_val_global"),
    "adj": ("p_adj_specificity", "p_adj_global"),
}

# Metric columns to coerce from CSV strings to float (when present).
_NUMERIC_COLS = (
    "z_score_global", "p_val_global", "p_adj_global",
    "z_score_specificity", "p_val_specificity", "p_adj_specificity",
    "n_perturbed",
)


def classify_quadrant(p_specificity, p_global, p_cutoff=0.05):
    """Four-quadrant label from the two MC-spatial axis p-values.

    Re-implements mc_spatial/visualize.py::classify_guide: sig := p < p_cutoff (strict).
    """
    sig_x = float(p_specificity) < p_cutoff
    sig_y = float(p_global) < p_cutoff
    if sig_x and sig_y:
        return BOTH
    if sig_x:
        return X_ONLY
    if sig_y:
        return Y_ONLY
    return INERT


def load_dual_metrics(csv_path, p_cutoff=0.05, p_mode="raw"):
    """Read a *_Dual_Metrics.csv into a list of dict rows, each with a `quadrant` label.

    The `guide` field is copied to `perturbation` (the benchmark scoring key) and the original
    `guide` key is dropped. `p_mode` selects which p-values drive classification: "raw" (p_val_*)
    or "adj" (p_adj_*). Numeric metric columns are coerced to float for downstream reporting.
    """
    if p_mode not in _P_COLS:
        raise ValueError(f"p_mode must be one of {sorted(_P_COLS)}, got {p_mode!r}")
    p_col_x, p_col_y = _P_COLS[p_mode]

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        if "guide" not in header:
            raise ValueError(f"{csv_path}: expected a 'guide' column, found {header}")
        missing = [c for c in (p_col_x, p_col_y) if c not in header]
        if missing:
            raise ValueError(
                f"{csv_path}: missing p-value columns for p_mode={p_mode!r}: {missing}"
            )
        rows = list(reader)

    out = []
    for r in rows:
        rec = {k: v for k, v in r.items() if k != "guide"}
        for col in _NUMERIC_COLS:
            if col in rec and rec[col] not in (None, ""):
                rec[col] = float(rec[col])
        rec["perturbation"] = r["guide"]
        rec["quadrant"] = classify_quadrant(r[p_col_x], r[p_col_y], p_cutoff=p_cutoff)
        out.append(rec)
    return out


def join_quadrants(records, dual_csv, key="perturbation", p_cutoff=0.05, p_mode="raw"):
    """Attach quadrant labels from a *_Dual_Metrics.csv onto benchmark scoring records (left join).

    `records` is an iterable of dicts each carrying `key` (default "perturbation"). Records whose
    key is absent from the CSV get quadrant = "Unknown". Returns NEW dicts (inputs are not
    mutated); only `quadrant` is added, so scoring fields are never overwritten.
    """
    dual = load_dual_metrics(dual_csv, p_cutoff=p_cutoff, p_mode=p_mode)
    label_of = {d["perturbation"]: d["quadrant"] for d in dual}
    out = []
    for rec in records:
        new = dict(rec)
        new["quadrant"] = label_of.get(rec[key], UNKNOWN)
        out.append(new)
    return out
