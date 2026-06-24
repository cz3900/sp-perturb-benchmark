# Three-board plots (seed unified + niche two-tier) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Make `plot_seed_prop` render **three boards** per the eval framework: (A) seed — all models, no upper bound; (B) niche · seed-only+Gaussian, upper bound = cell-1 (GT+Gaussian); (C) niche · end-to-end (GCN-mock), upper bound = cell-2 (GT+GCN). Per-prop upper bounds (not one shared line); drop the old `obs.mean+control` oracle. log(E-distance), lower=better.

**Architecture:** Everything is already in `res` — `fill_2x2` computes the four 2x2 cells (`GT+base`/`GT+learned`/`model+base`/`model+learned`) plus `null`. Keep `collect_seed_samples` (seed, no upper bound). Add `collect_niche_tier(res, tier)` that, for `tier='base'`, boxes `model+base` with floor=`null` and upper=`GT+base` (cell-1); for `tier='learned'`, boxes `model+learned` with floor=`null` and upper=`GT+learned` (cell-2). `plot_seed_prop` becomes 3 panels. Reference: `model/benchmark_评测框架_两类model_2026-06-25.md`.

**Tech Stack:** Python 3.11, numpy, matplotlib(Agg), pytest, existing `spbench`.

---

## File Structure
- `spbench/plotting.py` (modify): add `NICHE_TIERS` map + `collect_niche_tier(res, tier)`; rewrite `plot_seed_prop` to 3 panels; update `_DASH_COLORS` keys (`null (floor)`, `GT-seed (upper)`); keep `collect_seed_samples`. Drop `collect_prop_samples`' `oracle` dashed (and the function may stay for back-compat but is no longer used by `plot_seed_prop`).
- `tests/test_plotting.py` (modify): the existing `test_collect_prop_samples_*` / `test_plot_seed_prop_*` need updating to the new 3-board API + add discriminating asserts (seed has NO upper; tier1 upper == cell-1 `GT+base`; tier2 upper == cell-2 `GT+learned`; no `oracle` dashed).

---

### Task 1: three-board collect + plot

**Files:**
- Modify: `spbench/plotting.py`
- Test: `tests/test_plotting.py`

- [ ] **Step 1: Write/adjust the failing tests.** Replace the existing niche/plot tests with these (keep the imports + `GCN_KW` at top of the file; keep `test_collect_seed_samples`):

```python
def test_collect_niche_tier_base_has_cell1_upper(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    boxes, dashed = collect_niche_tier(res, "base")
    assert any("Gaussian" in k for k in boxes)          # seed-only + Gaussian box
    assert "null (floor)" in dashed
    assert "GT-seed (upper)" in dashed                  # cell-1 = GT+base
    # upper equals mean cell-1 energy, NOT the dropped obs.mean oracle
    import numpy as np
    exp = np.nanmean([c["e"]["GT+base"] for c in res["compare"].values()])
    assert np.isclose(dashed["GT-seed (upper)"], float(exp))
    assert "oracle" not in dashed

def test_collect_niche_tier_learned_has_cell2_upper(synth):
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    boxes, dashed = collect_niche_tier(res, "learned")
    assert any("GCN" in k for k in boxes)               # end-to-end mock box
    import numpy as np
    exp = np.nanmean([c["e"]["GT+learned"] for c in res["compare"].values()])
    assert np.isclose(dashed["GT-seed (upper)"], float(exp))   # cell-2 = GT+learned

def test_plot_seed_prop_three_boards(synth):
    import matplotlib; matplotlib.use("Agg")
    res = run_benchmark(synth, perturbations=["P0"], gcn_kwargs=GCN_KW, progress=False)
    fig = plot_seed_prop(res)
    assert len(fig.axes) == 3
    titles = [a.get_title() for a in fig.axes]
    assert any("seed" in t for t in titles)
    assert sum("niche" in t for t in titles) == 2
    # seed board (axes[0]) has NO upper-bound dashed line -> its legend has only null (or none)
    seed_leg = fig.axes[0].get_legend()
    seed_labels = [t.get_text() for t in seed_leg.get_texts()] if seed_leg else []
    assert "GT-seed (upper)" not in seed_labels
```

Also update `collect_niche_tier` and `plot_seed_prop` to the imports at the top of the test file (add `collect_niche_tier`). Remove or rewrite the old `test_collect_prop_samples_has_named_gcn` and `test_plot_seed_prop_returns_two_axes_with_gcn` (they assert the OLD 2-axis API and `oracle`).

- [ ] **Step 2: Run, confirm FAIL.** `cd /Users/cz/Documents/ZengLab/model/sp-perturb-benchmark && source .venv/bin/activate && python -m pytest tests/test_plotting.py -q` → FAIL (`collect_niche_tier` undefined / 3-axis assert).

- [ ] **Step 3: Implement.** In `spbench/plotting.py`:
  - Change `_DASH_COLORS` to `{"null (floor)": "#888888", "GT-seed (upper)": "#1d9e75"}`.
  - Add after `PROP_LABELS`:
```python
# niche board per prop tier: (box label, box 2x2 cell, upper-bound 2x2 cell = GT seed + that prop)
NICHE_TIERS = {
    "base":    ("Gaussian (seed-only)",   "model+base",    "GT+base"),     # tier 1: seed-only + Gaussian, upper = cell-1
    "learned": ("GCN (end-to-end mock)",  "model+learned", "GT+learned"),  # tier 2: end-to-end mock, upper = cell-2
}


def collect_niche_tier(res, tier):
    """One niche board: box = pooled per-repeat energy of the tier's model cell; dashed =
    null (floor) and GT-seed (per-prop upper bound = GT seed + that prop)."""
    label, box_cell, upper_cell = NICHE_TIERS[tier]
    cmp = res.get("compare", {})
    pooled = [s for c in cmp.values() for s in c.get("e_samples", {}).get(box_cell, [])]
    boxes = {label: np.asarray(pooled, float)} if pooled else {}
    dashed = {}
    nulls = [c["e"]["null"] for c in cmp.values() if "null" in c.get("e", {})]
    uppers = [c["e"][upper_cell] for c in cmp.values() if upper_cell in c.get("e", {})]
    if nulls:
        dashed["null (floor)"] = float(np.nanmean(nulls))
    if uppers:
        dashed["GT-seed (upper)"] = float(np.nanmean(uppers))
    return boxes, dashed
```
  - Update `collect_seed_samples`'s `dashed` key from `"null"` to `"null (floor)"` (so the colour map matches; it has no upper bound).
  - Rewrite `plot_seed_prop`:
```python
def plot_seed_prop(res, figsize=(15, 4.3)):
    """Three boards: seed (all models, floor only), niche seed-only+Gaussian (upper=cell-1),
    niche end-to-end/GCN-mock (upper=cell-2). log(E-distance), lower=better."""
    import matplotlib.pyplot as plt
    sb, sd = collect_seed_samples(res)
    b1, d1 = collect_niche_tier(res, "base")
    b2, d2 = collect_niche_tier(res, "learned")
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)
    _draw_boxes(ax0, sb, sd, "log E-distance", "seed — all models (floor = null)")
    _draw_boxes(ax1, b1, d1, "log E-distance", "niche · seed-only + Gaussian")
    _draw_boxes(ax2, b2, d2, "log E-distance", "niche · end-to-end (GCN mock)")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run, confirm PASS.** `python -m pytest tests/test_plotting.py -q` → PASS. Then full suite `python -m pytest -q` → all green (fix any other test that imported the removed `oracle`/2-axis behavior; e.g. the demo smoke test still works since it only savefigs the fig).

- [ ] **Step 5: Commit.**
```bash
git add spbench/plotting.py tests/test_plotting.py
git commit -m "feat(plotting): three-board plot (seed unified + niche two-tier, per-prop upper bounds)"
```

---

### Task 2: notebook §5b + Saunders re-render (controller-run)

**Files:** `notebooks/build_notebook.py` (§5b print uses `collect_prop_samples`; update to the new tiers), regenerate, re-execute on server.

- [ ] Update §5b code cell's print lines from `collect_prop_samples` to: `from spbench.plotting import plot_seed_prop, collect_seed_samples, collect_niche_tier` and print the seed + two niche tiers' method labels.
- [ ] `python notebooks/build_notebook.py`; commit + push template.
- [ ] Sync to server (git archive + rsync .git), patch config, `nbconvert --execute`, scp executed notebook + the 3-board PNG back, commit.

---

## Self-Review
- **Coverage**: framework doc §2 three boards → Task 1 (collect_niche_tier base/learned + 3-panel plot); §3 per-prop upper (cell-1/cell-2) → the `NICHE_TIERS` upper_cell; drop obs.mean oracle → test asserts no `oracle` dashed; seed no upper → test asserts seed legend lacks `GT-seed (upper)`. Naming seed/niche → titles/labels.
- **Out of scope** (later): real scGEN/CONCERT in the boards (needs §6 merge into the seed board + extra), multiple seed-only models per niche-tier-1 board.
- **Type consistency**: dashed keys `"null (floor)"`/`"GT-seed (upper)"` match `_DASH_COLORS`; box cells `model+base`/`model+learned` and upper cells `GT+base`/`GT+learned` match `fill_2x2`'s `METHODS`.

## Execution
Subagent-driven (Opus). Task 1 → implementer + spec + quality review. Task 2 controller-run (server).
