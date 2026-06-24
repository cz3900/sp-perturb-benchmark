# Plan 4 — end-to-end / external models via `extra` Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Score an end-to-end / external niche model (CONCERT-style) on the SAME footing as the 2x2 cells and surface it in the figure. The `extra=` plumbing in `compare_to_baseline` and `ConcertModel.predict_niche` already exist + are tested; this plan adds (1) a one-call `score_external_niche`, (2) `run_benchmark(external_models=...)` threading so an external model's niche lands in `res['compare'][p]['e_samples'][name]`, (3) a `MockEndToEnd` proving the path end-to-end, (4) a plotting hook so external models appear as boxes in the end-to-end niche board.

**Architecture:** `EndToEndModel` ABC (`models/base.py`) has `fit(train, edges)` + `predict(...)`; the de-facto niche interface is `predict_niche(data, perturbation, edges) -> (n_bystanders, n_genes)` (ConcertModel implements it). `run_benchmark` already calls `compare_to_baseline(niches, ...)` per perturbation — thread `extra` there. The three-board plot's end-to-end board is `collect_niche_tier(res, "learned")`; extend it to also box any external (non-2x2/null/oracle) methods present in `e_samples`.

**Tech Stack:** Python 3.11, numpy, pytest, existing `spbench`.

---

## File Structure
- `spbench/external.py` (new): `score_external_niche(...)` standalone scorer.
- `spbench/models/mock_end_to_end.py` (new): `MockEndToEnd(EndToEndModel)` for tests/demo.
- `spbench/config.py` (modify): `run_benchmark(..., external_models=None)` threads `extra`.
- `spbench/plotting.py` (modify): `collect_niche_tier(res, "learned")` includes external boxes; add `external_methods(res)` helper.
- `tests/test_external_models.py` (new).

---

### Task 1: score_external_niche + MockEndToEnd

**Files:** new `spbench/external.py`, `spbench/models/mock_end_to_end.py`, test in `tests/test_external_models.py`.

- [ ] **Step 1 — failing tests** (`tests/test_external_models.py`):
```python
import numpy as np
from spbench.synthetic import make_synthetic
from spbench.graph import build_knn_graph
from spbench.external import score_external_niche
from spbench.models.mock_end_to_end import MockEndToEnd

def test_mock_end_to_end_beats_null():
    data = make_synthetic(0)
    edges = build_knn_graph(data, k=10)
    p = data.perturbations()[0]
    model = MockEndToEnd(noise=0.3, seed=0).fit(data, edges)
    niche_pred = model.predict_niche(data, p, edges)
    assert niche_pred.shape[1] == data.n_genes and len(niche_pred) > 0
    r = score_external_niche(data, p, edges, niche_pred, name="mock")
    assert "mock" in r["e"] and "mock" in r["e_samples"]
    assert np.isfinite(r["e"]["mock"])
    assert r["e"]["mock"] < r["e"]["null"]          # a near-observed model beats 'no effect'
```

- [ ] **Step 2 — confirm FAIL** (`score_external_niche`/`MockEndToEnd` undefined).

- [ ] **Step 3 — implement.**
  `spbench/models/mock_end_to_end.py`:
```python
import numpy as np
from .base import EndToEndModel
from ..graph import neighbors_of


class MockEndToEnd(EndToEndModel):
    """A stand-in end-to-end model for testing the external-model path: its predicted bystander
    niche is the OBSERVED bystander niche plus gaussian noise (so it sits near the oracle ceiling,
    clearly below the null). NOT a real model — it reads observed expression, which a real model may
    not. Use only to exercise plumbing/plots."""
    name = "mock_end_to_end"

    def __init__(self, noise=0.3, seed=0):
        self.noise = float(noise); self.seed = int(seed)

    def fit(self, train, edges):
        return self

    def _bystanders(self, data, perturbation, edges):
        centers = np.where(data.perturbation == perturbation)[0]
        nbs = [neighbors_of(c, edges) for c in centers]
        nbs = [nb[~data.is_perturbed[nb]] for nb in nbs]
        return np.concatenate(nbs) if any(len(x) for x in nbs) else np.array([], int)

    def predict_niche(self, data, perturbation, edges):
        idx = self._bystanders(data, perturbation, edges)
        if len(idx) == 0:
            return np.zeros((0, data.n_genes), float)
        rng = np.random.default_rng(self.seed)
        return data.X[idx] + rng.normal(scale=self.noise, size=(len(idx), data.n_genes))

    def predict(self, perturbation, reference_cells, edges, center, neighbors):
        raise NotImplementedError("MockEndToEnd only implements predict_niche (niche scoring path).")
```
  `spbench/external.py`:
```python
import numpy as np
from .propagation_gt import propagation_gt
from .compare import compare_to_baseline


def score_external_niche(data, perturbation, edges, niche_pred, name="external",
                         residuals=None, eval_X=None, k_ref=5, repeats=20, seed=0, max_n=300):
    """Score an external/end-to-end model's predicted bystander niche on the same matched-n energy
    / gain / PCC-delta footing as the 2x2 cells. Builds the observed + no-effect-reference niches
    via propagation_gt and passes `niche_pred` through compare_to_baseline(extra={name: niche_pred}).
    Returns the compare_to_baseline dict (so res['e'][name], res['e_samples'][name], etc.)."""
    gt = propagation_gt(data, perturbation, edges, k_ref=k_ref)
    niches = {"observed": gt["perturbed_niche"], "reference": gt["reference_niche"]}
    return compare_to_baseline(niches, residuals=residuals, repeats=repeats, seed=seed,
                               max_n=max_n, extra={name: np.asarray(niche_pred, float)}, eval_X=eval_X)
```
  Note: `compare_to_baseline` reads `niches["1".."4"]` only `if k in niches` (it tolerates their absence — verify in compare.py); with just observed/reference/extra it scores null + the external method. If it requires the 2x2 keys, pass them as copies of reference (documented) — but first check; the loop is `for k in METHODS if k in niches`, so absence is fine.

- [ ] **Step 4 — PASS.** `python -m pytest tests/test_external_models.py -q`.

---

### Task 2: run_benchmark(external_models=...)

**Files:** modify `spbench/config.py`; test in `tests/test_external_models.py`.

- [ ] **Step 1 — failing test:**
```python
def test_run_benchmark_threads_external_models():
    from spbench.config import run_benchmark
    from spbench.models.mock_end_to_end import MockEndToEnd
    data = make_synthetic(0)
    p = data.perturbations()[0]
    res = run_benchmark(data, perturbations=[p], k=10, k_ref=5,
                        gcn_kwargs={"hidden": 8, "epochs": 3}, progress=False,
                        external_models={"mock": MockEndToEnd(noise=0.3, seed=0)})
    c = res["compare"][p]
    assert "mock" in c["e"] and "mock" in c["e_samples"]
    assert c["e"]["mock"] < c["e"]["null"]
```

- [ ] **Step 2 — implement.** In `run_benchmark`, add param `external_models=None`. In the per-perturbation loop where `compare and "_niches" in g`, build `extra = {nm: m.predict_niche(data, p, edges) for nm, m in external_models.items()} if external_models else None` and pass `extra=extra` to `compare_to_baseline(...)`. Default None → behavior unchanged.

- [ ] **Step 3 — PASS + confirm default path unchanged** (existing config tests green).

---

### Task 3: plotting — external models in the end-to-end board

**Files:** modify `spbench/plotting.py`; test in `tests/test_external_models.py`.

- [ ] **Step 1 — failing test:**
```python
def test_plot_includes_external_box():
    import matplotlib; matplotlib.use("Agg")
    from spbench.config import run_benchmark
    from spbench.models.mock_end_to_end import MockEndToEnd
    from spbench.plotting import collect_niche_tier
    data = make_synthetic(0)
    p = data.perturbations()[0]
    res = run_benchmark(data, perturbations=[p], k=10, k_ref=5,
                        gcn_kwargs={"hidden": 8, "epochs": 3}, progress=False,
                        external_models={"mock": MockEndToEnd(noise=0.3, seed=0)})
    boxes, dashed = collect_niche_tier(res, "learned")
    assert any("mock" in k for k in boxes)        # external model shows as its own box on the end-to-end board
```

- [ ] **Step 2 — implement.** Add to `plotting.py`:
```python
_STD_METHODS = {"GT+base", "GT+learned", "model+base", "model+learned", "null", "oracle"}

def external_methods(res):
    """Names of external/end-to-end methods present in res['compare'] (anything in e_samples that
    is not a standard 2x2 cell / null / oracle)."""
    cmp = res.get("compare", {})
    names = set()
    for c in cmp.values():
        names |= (set(c.get("e_samples", {})) - _STD_METHODS)
    return sorted(names)
```
  Extend `collect_niche_tier(res, tier)`: for `tier == "learned"`, after building the GCN box, also add one box per `external_methods(res)` (pooled per-repeat `e_samples[name]`), labelled by the method name. Keep dashed (null floor + GT-seed upper) unchanged. (For `tier == "base"` do not add externals.)

- [ ] **Step 3 — PASS + full suite + commit.** `python -m pytest -q` green. `git add -A && git commit -m "feat(external): score end-to-end models via run_benchmark external_models + score_external_niche + MockEndToEnd + plot in end-to-end board (Plan 4)"`

---

## Self-Review
- **Coverage**: roadmap Plan 4 — `score_external_niche` ✓ (Task 1), run_benchmark threading ✓ (Task 2), mock appears in `res` + niche figure ✓ (Tasks 2-3), gain between null and oracle → `e[mock] < e[null]` asserted (near-oracle). CONCERT real接入 = ConcertModel (already exists) passed as an external_model with its offline `.h5ad` paths — no new code needed, data-gated.
- **Non-breaking**: `external_models=None` default leaves run_benchmark unchanged; `collect_niche_tier` only adds boxes when externals exist.
- **Reuse**: `compare_to_baseline(extra=)`, `propagation_gt`, `EndToEndModel`, `collect_niche_tier` — no metric/niche reinvention.
