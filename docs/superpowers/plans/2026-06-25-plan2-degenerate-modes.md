# Plan 2 — degenerate modes (no cell_type / no NTC) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Make the aggregate-control reference robust to two degenerate datasets: (a) a single cell-line (one `cell_type` value → reference degenerates to all control cells); (b) no NTC (`'control'`) cells → fall back to `'none'` (unlabeled) cells as the control pool. `run_benchmark` must not crash and produce non-NaN seed/niche scores.

**Architecture:** Add a `control_pool` boolean-mask property on `StandardData`: `is_control` when present, else `is_unlabeled`, else all cells. Swap every place that uses `data.is_control` *as the control reference pool* to `data.control_pool` (NOT `is_perturbed`, which must stay literal). Single-cell-type already degenerates in `control_reference_centers` (same-type filter selects all controls); add a discriminating test to lock it.

**Tech Stack:** Python 3.11, numpy, pytest, existing `spbench`.

---

## File Structure
- `spbench/data.py` (modify): add `control_pool` + `has_ntc` properties on `StandardData`.
- `spbench/reference_aggregate.py` (modify): `aggregate_control` + `control_reference_centers` use `data.control_pool`.
- `spbench/harness.py` (modify): `_control_reference` + `_control_residuals` use `data.control_pool`.
- `tests/test_degenerate_modes.py` (new): 4 discriminating tests.

---

### Task 1: control_pool fallback + degenerate-mode robustness

**Files:** modify `spbench/data.py`, `spbench/reference_aggregate.py`, `spbench/harness.py`; new `tests/test_degenerate_modes.py`.

- [ ] **Step 1 — failing tests.** New `tests/test_degenerate_modes.py`. Build small synthetic `StandardData` directly (match how other tests construct it — check an existing test that imports `StandardData`). Three scenarios: (A) has 'control' cells; (B) NO 'control', has 'none' cells + gene-symbol perturbed cells; (C) single `cell_type` value with 'control' cells.

```python
import numpy as np, pytest
from spbench.data import StandardData
from spbench.reference_aggregate import control_reference_centers, aggregate_control
from spbench.graph import build_knn_graph
from spbench.config import run_benchmark

def _data(perts, cts, n_genes=12, seed=0):
    rng = np.random.default_rng(seed)
    n = len(perts)
    X = rng.normal(size=(n, n_genes)).astype(float)
    coords = rng.normal(size=(n, 2))
    return StandardData(X=X, coords=coords, perturbation=np.array(perts),
                        cell_type=np.array(cts), batch=np.array(["b"]*n),
                        gene_names=[f"g{i}" for i in range(n_genes)])

def test_control_pool_uses_control_when_present():
    d = _data(["control","control","P0","P0"], ["A","A","A","A"])
    assert d.has_ntc is True
    assert np.array_equal(d.control_pool, d.is_control)

def test_control_pool_falls_back_to_none_when_no_ntc():
    d = _data(["none","none","P0","P0"], ["A","A","A","A"])
    assert d.has_ntc is False
    assert np.array_equal(d.control_pool, d.is_unlabeled)   # 'none' cells become the control pool

def test_single_cell_type_degenerates_to_all_controls():
    # 4 controls of the SAME (only) cell type + 2 perturbed
    d = _data(["control"]*4 + ["P0","P0"], ["A"]*6)
    centers = np.where(d.perturbation == "P0")[0]
    refs = control_reference_centers(d, centers)
    ctrl_idx = np.where(d.is_control)[0]
    for r in refs:
        assert np.array_equal(np.sort(r), np.sort(ctrl_idx))   # every center's reference = ALL controls

def test_run_benchmark_no_ntc_non_nan():
    # bigger synthetic so the graph/energy run; no 'control', use 'none' as pool
    rng = np.random.default_rng(1)
    n = 60
    perts = np.array((["none"]*30) + (["P0"]*30))
    cts = np.array(["A"]*n)
    X = rng.normal(size=(n, 12)); X[perts=="P0"] += 1.5   # planted seed shift
    coords = rng.normal(size=(n, 2))
    d = StandardData(X=X, coords=coords, perturbation=perts, cell_type=cts,
                     batch=np.array(["b"]*n), gene_names=[f"g{i}" for i in range(12)])
    res = run_benchmark(d, perturbations=["P0"], k=8, k_ref=4,
                        gcn_kwargs={"hidden": 8, "epochs": 3}, progress=False)
    s = res["seed"]["P0"]; c = res["compare"]["P0"]
    assert np.isfinite(s["pcc_delta"])           # seed score computed from the 'none' control pool
    assert np.isfinite(c["e"]["model+base"])     # niche score non-NaN
```

- [ ] **Step 2 — confirm FAIL.** `python -m pytest tests/test_degenerate_modes.py -q` → fails (`control_pool`/`has_ntc` undefined; no-NTC run may crash/NaN).

- [ ] **Step 3 — implement.**
  - `spbench/data.py`, add to `StandardData`:
```python
    @property
    def control_pool(self) -> np.ndarray:
        """Boolean mask of cells used as the unperturbed CONTROL reference: the 'control' (NTC)
        cells when present; else fall back to 'none' (unlabeled) cells (datasets without NTC, e.g.
        Shen); else all cells (degenerate)."""
        if self.is_control.any():
            return self.is_control
        if self.is_unlabeled.any():
            return self.is_unlabeled
        return np.ones(self.n_cells, dtype=bool)

    @property
    def has_ntc(self) -> bool:
        return bool(self.is_control.any())
```
  - `spbench/reference_aggregate.py`: in `aggregate_control`, change `ctrl = data.is_control` → `ctrl = data.control_pool` (the gmean line and the per-ct control filtering then use this pool). In `control_reference_centers`, change `ctrl_idx = np.where(data.is_control)[0]` → `np.where(data.control_pool)[0]`. Keep `min_control`/global fallback logic.
  - `spbench/harness.py`: in `_control_reference` change `ctrl = data.is_control` → `data.control_pool`; in `_control_residuals` change `ctrl = data.is_control` → `data.control_pool`. Do NOT touch `is_perturbed` or `_bystanders` (perturbed set stays literal; bystanders = non-perturbed, which already includes the 'none'/control pool).

- [ ] **Step 4 — confirm PASS + full suite.** `python -m pytest tests/test_degenerate_modes.py -q` PASS; then `python -m pytest -q` all green (the existing Saunders-shaped tests still pass since `control_pool == is_control` whenever controls exist).

- [ ] **Step 5 — commit.** `git add -A && git commit -m "feat(data): control_pool fallback to 'none' when no NTC + single-cell-type degeneration (Plan 2)"`

---

## Self-Review
- **Coverage**: roadmap Plan 2 ① single cell_type → `test_single_cell_type_degenerates`; ② no-NTC → `control_pool` fallback + `test_run_benchmark_no_ntc_non_nan`. Change points (reference_aggregate ×2, harness ×2) all swapped.
- **Non-breaking**: `control_pool == is_control` whenever any control exists, so all existing behavior/tests are unchanged.
- **Out of scope**: adapters that produce these degenerate datasets (Cheng=Plan 7, Shen=Plan 8).
