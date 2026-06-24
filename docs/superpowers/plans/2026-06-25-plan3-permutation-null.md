# Plan 3 — permutation null (empirical null + p) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.
> NOTE: the kNN-SP/kNN-GEX baselines originally bundled here are DEFERRED — they don't compose with the distributional energy metric (a copied-niche prediction matches the observed distribution → energy≈0). Saved as a question for the user (`model/明早问_2026-06-25.md` Q1). This plan delivers the permutation null only.

**Goal:** Give each perturbation an empirical null for its niche shift: draw `|centers|` random non-perturbed cells as fake centers, measure their bystander-niche energy vs the no-effect reference niche, repeat `n_perm` times; report the real perturbation's niche energy and its empirical p (fraction of null ≥ real). `run_benchmark` exposes it optionally; an inert perturbation → high p, a planted one → low p.

**Architecture:** New `spbench/permutation.py::permutation_null` reuses `propagation_gt` (perturbed_niche, reference_niche, centers) + `_bystander_neighbors` + `metrics.energy.energy_distance`, with matched-n subsampling (same bias-cancellation idea as `compare`). `config.run_benchmark` gains an optional `n_perm` kwarg; when set it stores `res["perm"][p] = {null, real, p}`.

**Tech Stack:** Python 3.11, numpy, pytest, existing `spbench`.

---

## File Structure
- `spbench/permutation.py` (new): `permutation_null(data, perturbation, edges, n_perm=50, seed=0, max_n=300)`.
- `spbench/config.py` (modify): `run_benchmark(..., n_perm=None)`; when not None, compute + store `res["perm"]`.
- `tests/test_permutation_null.py` (new): planted→low p, inert→high p, empty→NaN.

---

### Task 1: permutation_null module

**Files:** new `spbench/permutation.py`, `tests/test_permutation_null.py`.

- [ ] **Step 1 — failing test.** New `tests/test_permutation_null.py`:

```python
import numpy as np
from spbench.data import StandardData
from spbench.graph import build_knn_graph
from spbench.permutation import permutation_null

def _spatial(n_per=40, planted=True, seed=0):
    """Two spatial blobs of cells. Half perturbed (P0), half control. When planted=True the
    bystander neighbours of perturbed centers get a niche shift; otherwise no shift (inert)."""
    rng = np.random.default_rng(seed)
    n = n_per * 3
    coords = rng.normal(size=(n, 2)) * 5
    perturbation = np.array(["control"]*n)
    # make a third of cells perturbed centers spread in space
    cen = rng.choice(n, n_per, replace=False)
    perturbation[cen] = "P0"
    X = rng.normal(size=(n, 10)).astype(float)
    d = StandardData(X=X, coords=coords, perturbation=perturbation,
                     cell_type=np.array(["A"]*n), batch=np.array(["b"]*n),
                     gene_names=[f"g{i}" for i in range(10)])
    edges = build_knn_graph(d, k=8)
    if planted:
        # shift the bystander neighbours of perturbed centers
        from spbench.graph import neighbors_of
        hit = set()
        for c in cen:
            for j in neighbors_of(c, edges):
                if perturbation[j] != "P0":
                    hit.add(int(j))
        idx = np.array(sorted(hit))
        if len(idx):
            X[idx] += 3.0
    return d, edges

def test_planted_perturbation_has_low_p():
    d, edges = _spatial(planted=True, seed=1)
    r = permutation_null(d, "P0", edges, n_perm=40, seed=0)
    assert np.isfinite(r["real"]) and r["null"]
    assert r["p"] <= 0.2          # real niche shift stands out from random patches

def test_inert_perturbation_has_high_p():
    d, edges = _spatial(planted=False, seed=2)
    r = permutation_null(d, "P0", edges, n_perm=40, seed=0)
    assert r["p"] >= 0.2          # no shift -> indistinguishable from background

def test_empty_perturbation_returns_nan():
    d, edges = _spatial(planted=False, seed=3)
    r = permutation_null(d, "ABSENT", edges, n_perm=10, seed=0)
    assert np.isnan(r["p"])
```
(If `build_knn_graph`'s signature differs, check `spbench/graph.py` and adapt the call; keep the asserts.)

- [ ] **Step 2 — confirm FAIL.** `python -m pytest tests/test_permutation_null.py -q` → `permutation_null` undefined.

- [ ] **Step 3 — implement `spbench/permutation.py`:**
```python
import numpy as np
from .propagation_gt import propagation_gt, _bystander_neighbors
from .metrics.energy import energy_distance


def _matched_energy(A, B, rng, max_n):
    A = np.asarray(A, float); B = np.asarray(B, float)
    if len(A) < 1 or len(B) < 1:
        return float("nan")
    n = max(2, min(len(A), len(B), max_n))
    a = A[rng.choice(len(A), n, replace=False)] if len(A) > n else A
    b = B[rng.choice(len(B), n, replace=False)] if len(B) > n else B
    return energy_distance(a, b)


def permutation_null(data, perturbation, edges, n_perm=50, seed=0, max_n=300):
    """Empirical null for a perturbation's niche shift. real = matched-n energy(perturbed niche,
    reference niche). null_i = energy(bystander niche of |centers| random non-perturbed 'fake'
    centers, reference niche). p = (#{null >= real} + 1)/(len(null)+1). Low p: the perturbation
    shifts its niche more than random patches; high p (inert): indistinguishable from background."""
    gt = propagation_gt(data, perturbation, edges)
    perturbed, reference, centers = gt["perturbed_niche"], gt["reference_niche"], gt["centers"]
    if len(perturbed) == 0 or len(reference) == 0 or len(centers) == 0:
        return {"null": [], "real": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(seed)
    real = _matched_energy(perturbed, reference, rng, max_n)
    pool = np.where(~data.is_perturbed)[0]
    null = []
    k = min(len(centers), len(pool))
    for _ in range(n_perm):
        fake = rng.choice(pool, k, replace=False)
        nb = [_bystander_neighbors(data, c, edges) for c in fake]
        nb = np.concatenate(nb) if any(len(x) for x in nb) else np.array([], int)
        if len(nb) == 0:
            continue
        null.append(_matched_energy(data.X[nb], reference, rng, max_n))
    null = [x for x in null if np.isfinite(x)]
    p = (np.sum(np.asarray(null) >= real) + 1) / (len(null) + 1) if null else float("nan")
    return {"null": null, "real": float(real), "p": float(p)}
```

- [ ] **Step 4 — confirm PASS.** `python -m pytest tests/test_permutation_null.py -q` PASS.

- [ ] **Step 5 — wire into run_benchmark.** In `spbench/config.py::run_benchmark`, add kwarg `n_perm=None`. After the per-perturbation seed/compare loop (where `res` is assembled — match the existing structure), when `n_perm` is not None compute `res["perm"] = {p: permutation_null(data, p, edges, n_perm=n_perm, seed=seed) for p in perturbations}` (use the same `edges` the benchmark built; check the variable name in run_benchmark). Import `permutation_null` at top. Add one test:
```python
def test_run_benchmark_exposes_perm(...):
    # reuse a small synthetic with a planted effect; run_benchmark(..., n_perm=20)
    # assert 'perm' in res and 0 <= res['perm'][p]['p'] <= 1
```
Place it in `tests/test_permutation_null.py` using the same synthetic builder + `run_benchmark` (check its signature in `spbench/config.py`).

- [ ] **Step 6 — full suite + commit.** `python -m pytest -q` all green. `git add -A && git commit -m "feat(permutation): empirical permutation null + p-value, wired into run_benchmark (Plan 3 core)"`

---

## Self-Review
- **Coverage**: roadmap Plan 3 permutation null → module + run_benchmark `n_perm` + planted/inert p test. kNN-SP/GEX deferred (saved question, metric mismatch with energy distance).
- **Reuse**: `propagation_gt` + `_bystander_neighbors` + `energy_distance` (no metric reinvention); matched-n mirrors `compare`.
- **Out of scope**: plotting the permutation percentile line (optional; p-value in `res` suffices for now); kNN baselines (deferred).
