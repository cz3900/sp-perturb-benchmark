# Unify seed/niche on one delta metric Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Make seed and niche use ONE logic: **PCC-delta = primary** (mean-shift direction, bounded, cross-dataset comparable), **E-distance = secondary** (both with control residuals so variance is aligned and E reflects the mean shift, not the variance collapse), with **null/oracle references** on both. Fixes the seed bug (energy scored a collapsed mean-field cloud without residuals → null trivially won). One coherent figure: seed | niche side by side, same axes.

**Architecture:** Niche predictions already get a control residual in `fill_2x2.collect()` (harness:126). Seed predictions don't — `seed_pred` is the raw per-center mean-field. Add a variance-restored `seed_pred_resid` in `fill_2x2` (residual only in the data.X space, i.e. when `eval_X` is NOT an ndarray — scGEN's log-norm per-center predictions already carry variance and live in a different space). `evaluate_seed` then scores **energy on `seed_pred_resid`** but keeps **pcc_delta/mse on the raw `seed_pred`** (mean-based, residual-invariant). Plotting gains a PCC-delta primary view; the existing E box-plot becomes the variance-aligned secondary.

**Tech Stack:** Python 3.11, numpy, matplotlib(Agg), pytest, existing `spbench`.

---

## File Structure
- `spbench/harness.py` (modify): `fill_2x2` adds `_niches["seed_pred_resid"]` (residual-restored seed, data.X space only).
- `spbench/compare.py` (modify): `evaluate_seed` scores energy on `seed_pred_resid`, pcc/mse on raw `seed_pred`.
- `spbench/plotting.py` (modify): add `collect_delta(res, kind)` + `plot_delta(res)` (PCC-delta, seed|niche). Keep `plot_seed_prop` (E box) — seed E now variance-aligned.
- `tests/test_unify_delta.py` (new): seed residual restores variance + energy drops to ~null + pcc unchanged; plot_delta has 2 panels.

---

### Task 1: seed energy gets control residuals (the bug fix)

**Files:** modify `spbench/harness.py`, `spbench/compare.py`; test `tests/test_unify_delta.py`.

- [ ] **Step 1 — failing test.** New `tests/test_unify_delta.py`:
```python
import numpy as np
from spbench.data import StandardData
from spbench.graph import build_knn_graph
from spbench.harness import fill_2x2, _control_reference_aggregate, _control_residuals
from spbench.models.trivial_seed import TrivialSeed
from spbench.models.gaussian_prop import GaussianProp
from spbench.models.gcn_prop import SimpleGCN
from spbench.compare import evaluate_seed

def _planted(n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    # one cell type, one guide P0 with a real mean shift + big cell-to-cell variance
    n = n_per * 3
    pert = np.array((["control"] * (2 * n_per)) + (["P0"] * n_per))
    X = rng.normal(scale=1.0, size=(n, 12))            # large per-cell variance (biological noise)
    X[pert == "P0"] += np.array([2.0] + [0.0] * 11)    # mean shift on gene 0
    d = StandardData(X=X, coords=rng.normal(size=(n, 2)), perturbation=pert,
                     cell_type=np.array(["A"] * n), batch=np.array(["b"] * n),
                     gene_names=[f"g{i}" for i in range(12)])
    return d

def _niches(d):
    edges = build_knn_graph(d, k=10)
    sm = TrivialSeed().fit(d); base = GaussianProp().fit(d, edges); lr = SimpleGCN(hidden=8, epochs=3).fit(d, edges)
    Xref = _control_reference_aggregate(d, edges); pool = _control_residuals(d)
    g = fill_2x2(d, "P0", edges, sm, base, lr, k_ref=5, X_ref=Xref, return_niches=True, residuals=pool)
    return g["_niches"]

def test_seed_pred_resid_restores_variance():
    n = _niches(_planted())
    sp = np.asarray(n["seed_pred"]); spr = np.asarray(n["seed_pred_resid"])
    assert sp.std(0).mean() < 1e-6                       # raw mean-field seed is collapsed
    assert spr.std(0).mean() > 0.3                       # residual restores per-cell variance
    assert np.allclose(sp.mean(0), spr.mean(0), atol=0.25)  # mean ~unchanged (zero-mean residual)

def test_seed_energy_fair_after_residual():
    n = _niches(_planted())
    r = evaluate_seed(n)                                  # no eval_X -> data.X space -> residual applies
    m = float(np.mean(r["e_samples"]["model"])); nul = float(np.mean(r["e_samples"]["null"]))
    # collapsed model used to be >> null; with residual it is in the same ballpark (within 2x)
    assert m < 2.0 * nul
    assert np.isfinite(r["pcc_delta"])                   # pcc still computed (on raw seed)

def test_pcc_delta_residual_invariant():
    n = _niches(_planted())
    # pcc_delta must be computed on the RAW seed_pred (mean), so injecting a resid key must not change it
    import copy
    r_raw = evaluate_seed({k: v for k, v in n.items() if k != "seed_pred_resid"})
    r_res = evaluate_seed(n)
    assert abs(r_raw["pcc_delta"] - r_res["pcc_delta"]) < 1e-9
```

- [ ] **Step 2 — confirm FAIL** (`seed_pred_resid` not in niches; energy still uses collapsed pred).

- [ ] **Step 3 — implement.**
  - `spbench/harness.py`, in `fill_2x2`, right after `seed_pred` is built (the `if not len(centers) / elif per_center_seed / else` block that sets `seed_pred`, ~line 143-149) and before the `seed_ref_idx` line, add:
```python
        # variance-restored seed for the ENERGY readout (pcc/mse stay on the raw mean seed):
        # mirror the niche path's control-residual so a collapsed mean-field seed isn't structurally
        # penalised by the energy distance. Only in data.X space — when eval_X is an ndarray (scGEN's
        # log-norm per-center predictions) the seed already carries variance in its own space, and the
        # data.X residual pool would be the wrong space, so skip it there.
        if residuals is not None and not isinstance(eval_X, np.ndarray) and len(seed_pred):
            _rng_s = np.random.default_rng(noise_seed + 7)
            seed_pred_resid = seed_pred + _draw_residuals(residuals, data.cell_type[centers], _rng_s)
        else:
            seed_pred_resid = seed_pred
```
    Then add `"seed_pred_resid": seed_pred_resid,` to the `grid["_niches"] = {...}` dict (next to `"seed_pred": seed_pred,`). (`_draw_residuals` is already defined in harness.py.)
  - `spbench/compare.py`, in `evaluate_seed`: after computing `pred`, add a variance-restored copy for energy:
```python
    pred_e = _apply_eval_X(niches.get("seed_pred_resid", niches.get("seed_pred", np.zeros((0, 0)))), eval_X)
```
    Change the energy clouds from `{"model": pred, "null": ref}` to `{"model": pred_e, "null": ref}`. Keep the `pcc_delta`/`mse` lines computing on the raw `pred` (unchanged). The empty-input guard stays on `pred` (raw).

- [ ] **Step 4 — PASS + full suite.** `python -m pytest tests/test_unify_delta.py -q` then `python -m pytest -q` all green.

- [ ] **Step 5 — commit.** `git add -A && git commit -m "fix(seed): control-residual variance restoration for seed energy (mirror niche); pcc/mse stay on raw mean"`

---

### Task 2: unified PCC-delta figure

**Files:** modify `spbench/plotting.py`; test `tests/test_unify_delta.py`.

- [ ] **Step 1 — failing test** (append):
```python
def test_plot_delta_two_panels():
    import matplotlib; matplotlib.use("Agg")
    from spbench.config import run_benchmark
    from spbench.plotting import plot_delta, collect_delta
    d = _planted()
    res = run_benchmark(d, perturbations=["P0"], k=10, k_ref=5,
                        gcn_kwargs={"hidden": 8, "epochs": 3}, progress=False)
    sb, _ = collect_delta(res, "seed"); nb, _ = collect_delta(res, "niche")
    assert sb and nb                                     # both have method->pcc lists
    fig = plot_delta(res)
    assert len(fig.axes) == 2
    titles = [a.get_title().lower() for a in fig.axes]
    assert any("seed" in t for t in titles) and any("niche" in t for t in titles)
```

- [ ] **Step 2 — implement** in `spbench/plotting.py`:
```python
def collect_delta(res, kind):
    """PCC-delta per method for one task. kind='seed' -> res['seed'][p]['pcc_delta'] (one box
    'seed model'); kind='niche' -> res['compare'][p]['pcc'][method] for the deployable cells
    (model+base, model+learned) + any external methods. Returns ({label: [pcc per guide]}, refs)
    where refs has 'perfect'=1.0 and 'no-corr'=0.0 reference levels (oracle pcc ~1, null pcc ~NaN)."""
    boxes = {}
    if kind == "seed":
        vals = [s["pcc_delta"] for s in res.get("seed", {}).values() if np.isfinite(s.get("pcc_delta", np.nan))]
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
    axes — the unified delta metric. Dashed: perfect=1, no-corr=0."""
    import matplotlib.pyplot as plt
    sb, sd = collect_delta(res, "seed")
    nb, nd = collect_delta(res, "niche")
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
```
  (Leave `plot_seed_prop` as the secondary E-distance view; with Task 1 its seed board is now variance-aligned.)

- [ ] **Step 3 — PASS + full suite + commit.** `python -m pytest -q` green. `git add -A && git commit -m "feat(plotting): plot_delta — unified PCC-delta primary view (seed | niche)"`

---

### Task 3: notebook + server re-render (controller-run)
- [ ] `build_notebook.py`: make PCC-delta the headline (base_df `seed_pcc`/`niche_pcc` primary; capability matrix uses PCC-delta seed/niche both higher=better; add `plot_delta` as the main figure, keep `plot_seed_prop` as the variance-aligned E supplement). Regenerate.
- [ ] Sync + execute on node03; confirm seed E now ~null-level (variance-aligned) and the PCC-delta figure renders; scp back + commit.

---

## Self-Review
- **Coverage**: seed residual fix (Task 1, the bug); unified PCC-delta primary + variance-aligned E secondary (Task 2); both have perfect/no-corr refs; notebook headline = PCC-delta (Task 3).
- **eval_X correctness**: residual only in data.X space (`not isinstance(eval_X, np.ndarray)`) so scGEN's log-norm path is untouched.
- **pcc invariance**: pcc/mse computed on raw `seed_pred`; `test_pcc_delta_residual_invariant` locks it.
- **Out of scope**: dropping the energy view entirely (kept as secondary), deeper covariance/distribution metrics.
