# Three-Model Reproduction (SpatialProp / SpaceTravLR / Celcomen) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three spatial end-to-end (D1 seed + D2 niche) perturbation models — SpatialProp, SpaceTravLR, Celcomen — to `spbench` as offline-dump external models, landing them on the seed board + niche class-2 board alongside CONCERT.

**Architecture:** Each model follows the proven offline-dump template: an **input adapter** (`StandardData → model h5`), an **offline runner** (model's own env + GPU, dumps a per-guide prediction `.h5ad`), and a **loader** (`EndToEndModel` reading the dump's predicted layer → bystander niche). Pure code is authored + unit-tested locally on synthetic data; the model-calling runner bodies and all real-data runs happen on the lab server (`srun` overlap into a Jupyter allocation). Off-panel KO handling = **option 2**: gene-column injectors (all three) run only in-panel guides; off-panel guides fall to floor and are covered by the seed-only path.

**Tech Stack:** Python, numpy, h5py, pytest (local); per-model server envs — SpatialProp `torch2.6/PyG2.6` conda, SpaceTravLR `uv` venv (celloracle/commot), Celcomen `python=3.9` conda.

---

## Server gate (read first)

The lab server (`zenglab`) is **currently unreachable from the dev machine (SSH/VPN down)**. All tasks tagged **[SERVER]** are blocked until the VPN is restored; **[LOCAL]** tasks (Phases 0, 2.1–2.4, 3.1–3.4) are fully actionable now and need no server, GPU, or real data — they run against `spbench.synthetic`. Do the LOCAL tasks first; queue the SERVER tasks.

Server run preamble (used by every [SERVER] task — discover the running Jupyter allocation and overlap into it):

```bash
JOBID=$(ssh zenglab "squeue -u \$(whoami) -h -o %i | head -1")
ssh zenglab "srun --jobid=$JOBID --overlap -w node03 bash -lc '
  export LD_LIBRARY_PATH=\$HOME/.conda/envs/concert/lib:\$LD_LIBRARY_PATH
  export PYTHONPATH=\$HOME/sp-perturb-benchmark:\$PYTHONPATH
  <command here>'"
```

Sync local code to the server (GitHub is blocked on the cluster — use git archive, never `git pull`):

```bash
git archive HEAD | ssh zenglab 'tar -x -C ~/sp-perturb-benchmark'
```

---

## Phase 0 — Shared pure utilities [LOCAL]

### Task 0.1: Guide-coverage helper

**Files:**
- Create: `spbench/coverage.py`
- Test: `tests/test_coverage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coverage.py
import pytest
from spbench.coverage import guide_overlap

def test_guide_overlap_splits_in_and_out():
    out = guide_overlap(["A", "B", "C"], ["B", "C", "D"])
    assert out["in"] == ["B", "C"]
    assert out["out"] == ["A"]

def test_guide_overlap_empty_allowed_all_out():
    out = guide_overlap(["A", "B"], [])
    assert out["in"] == [] and out["out"] == ["A", "B"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_coverage.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'spbench.coverage'`

- [ ] **Step 3: Write minimal implementation**

```python
# spbench/coverage.py
"""Split a dataset's guide genes into those a model can inject vs cannot, given the model's
injectable-gene set (panel genes for SpatialProp/Celcomen; TF∪ligand∪receptor for SpaceTravLR).
Off-panel guides are NOT dropped from the benchmark (they keep real GT marker shifts) — this only
tells the runner which guides a gene-column injector can represent (option 2)."""


def guide_overlap(guides, allowed):
    """guides, allowed: iterables of gene symbols. Returns {'in': sorted injectable guides,
    'out': sorted non-injectable guides}. Exact case-sensitive match (caller normalizes case)."""
    allowed = set(map(str, allowed))
    g = [str(x) for x in guides]
    return {"in": sorted([x for x in g if x in allowed]),
            "out": sorted([x for x in g if x not in allowed])}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_coverage.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add spbench/coverage.py tests/test_coverage.py
git commit -m "feat: guide-coverage helper (in/out-of-injectable-set split)"
```

### Task 0.2: Panel-alignment helper (for subset-gene predictions)

**Files:**
- Create: `spbench/dump_align.py`
- Test: `tests/test_dump_align.py`

SpaceTravLR/Celcomen may predict over a gene *subset* (HVGs ∩ GRN). Scoring needs predictions in the full panel-gene column order, with non-modeled genes left unchanged (= the unperturbed input, "no change"). This pure helper does that alignment; both new runners call it before dumping.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dump_align.py
import numpy as np, pytest
from spbench.dump_align import align_prediction_to_panel

def test_align_fills_unpredicted_from_fallback():
    pred = np.array([[10., 20.]])              # model predicted genes b, d only
    fallback = np.array([[1., 2., 3., 4.]])    # panel order a, b, c, d (unperturbed input)
    out = align_prediction_to_panel(pred, ["b", "d"], ["a", "b", "c", "d"], fallback)
    assert np.allclose(out, [[1., 10., 3., 20.]])

def test_align_row_mismatch_raises():
    with pytest.raises(ValueError):
        align_prediction_to_panel(np.zeros((2, 1)), ["a"], ["a", "b"], np.zeros((3, 2)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dump_align.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'spbench.dump_align'`

- [ ] **Step 3: Write minimal implementation**

```python
# spbench/dump_align.py
"""Map a model prediction over its own gene subset onto the benchmark panel column order, filling
genes the model did not predict from a fallback (the unperturbed input = 'no change'). Keeps the
loader/scoring on a single full-panel gene axis regardless of which genes the model modelled."""
import numpy as np


def align_prediction_to_panel(pred, pred_genes, panel_genes, fallback):
    """pred: (N, len(pred_genes)). fallback: (N, len(panel_genes)) unperturbed input in panel order.
    Returns (N, len(panel_genes)): fallback with each predicted gene's column overwritten by pred."""
    pred = np.asarray(pred, float); fallback = np.asarray(fallback, float)
    panel_genes = list(map(str, panel_genes)); pred_genes = list(map(str, pred_genes))
    if fallback.shape[1] != len(panel_genes):
        raise ValueError(f"fallback has {fallback.shape[1]} cols != {len(panel_genes)} panel genes")
    if pred.shape[0] != fallback.shape[0]:
        raise ValueError(f"pred has {pred.shape[0]} rows != fallback {fallback.shape[0]}")
    out = fallback.copy()
    idx = {g: j for j, g in enumerate(panel_genes)}
    for j, g in enumerate(pred_genes):
        if g in idx:
            out[:, idx[g]] = pred[:, j]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dump_align.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add spbench/dump_align.py tests/test_dump_align.py
git commit -m "feat: align subset-gene predictions onto the full panel axis"
```

---

## Phase 1 — SpatialProp finish [SERVER] (local code already done + tested)

`adapters/spatialprop_export.py`, `models/spatialprop_model.py`, `scripts/spatialprop/run_spatialprop.py`,
`scripts/spatialprop/ENV_SETUP.md` already exist and are covered by `tests/test_adapter_spatialprop.py`,
`tests/test_runner_spatialprop.py`, `tests/test_spatialprop_model.py`. Remaining work is server-only.

### Task 1.1: Confirm + patch the runner import name [SERVER]

`scripts/spatialprop/run_spatialprop.py:30` imports `from spatialprop import (...)`, but the repo
`abuendia/spatial-prop` package is `spatial_gnn` (`spatial_gnn.api.perturbation_api`). Confirm the real
import name in the built env, patch locally if needed, re-sync.

- [ ] **Step 1:** After Task 1.2 builds the env, probe the package name:

```bash
ssh zenglab "srun --jobid=$JOBID --overlap bash -lc '
  ~/.conda/envs/spatialprop/bin/python -c \"import spatial_gnn; print(spatial_gnn.__file__)\" || \
  ~/.conda/envs/spatialprop/bin/python -c \"import spatialprop; print(spatialprop.__file__)\"'"
```

- [ ] **Step 2:** If it resolves to `spatial_gnn`, patch the import locally (Edit `scripts/spatialprop/run_spatialprop.py:30`):

```python
    from spatial_gnn.api.perturbation_api import (train_perturbation_model,
        create_perturbation_input_matrix, predict_perturbation_effects)
```

- [ ] **Step 3:** Re-run the local runner test (import is lazy inside `main`, so it must still pass):

Run: `python -m pytest tests/test_runner_spatialprop.py -q`
Expected: PASS (2 passed)

- [ ] **Step 4: Commit + re-sync**

```bash
git add scripts/spatialprop/run_spatialprop.py
git commit -m "fix: spatialprop runner import path to spatial_gnn.api"
git archive HEAD | ssh zenglab 'tar -x -C ~/sp-perturb-benchmark'
```

### Task 1.2: Build the `spatialprop` conda env [SERVER]

- [ ] **Step 1:** Follow `scripts/spatialprop/ENV_SETUP.md` to create `~/.conda/envs/spatialprop` (torch2.6/PyG2.6) and `pip install` the `abuendia/spatial-prop` source (cloned to `~/spatial-prop`).
- [ ] **Step 2:** Smoke-check the API imports:

Run (via the server preamble): `~/.conda/envs/spatialprop/bin/python -c "from spatial_gnn.api.perturbation_api import train_perturbation_model; print('ok')"`
Expected: `ok`

### Task 1.3: Run SpatialProp on Saunders b10 [SERVER]

- [ ] **Step 1:** Export the slice (concert env, which has h5py + spbench deps):

```bash
~/.conda/envs/concert/bin/python -c "
from spbench.config import load_dataset
from spbench.adapters.counts_export import build_counts_X
from spbench.adapters.spatialprop_export import export_to_spatialprop_h5
d = load_dataset('saunders', path='~/saunders_b10_slice')
export_to_spatialprop_h5(d, build_counts_X(d), '/tmp/saunders_b10_spatialprop.h5ad')"
```
(If `load_dataset` is not the exact accessor, use the adapter registry `get_adapter('saunders')` as in `runs/verify_saunders.py`.)

- [ ] **Step 2:** For each in-panel guide `P`, run the runner into a dump dir:

```bash
mkdir -p ~/spatialprop_dumps_Saunders_b10
for P in $(in-panel guides from Task 0.1 coverage on saunders); do
  ~/.conda/envs/spatialprop/bin/python scripts/spatialprop/run_spatialprop.py \
    --h5ad /tmp/saunders_b10_spatialprop.h5ad --pert "$P" \
    --out ~/spatialprop_dumps_Saunders_b10/${P}.h5ad
done
```
Expected per guide: `wrote ...: predicted_tempered (N, G)`

### Task 1.4: Wire SpatialProp into the benchmark notebook [SERVER]

- [ ] **Step 1:** In `notebooks/run_benchmark.ipynb` set `SPATIALPROP_DUMP_DIR = "~/spatialprop_dumps_Saunders_b10"` and construct `SpatialPropModel({P: f"{DIR}/{P}.h5ad" for P in guides})`, passed via `external_models={"SpatialProp": model}`.
- [ ] **Step 2:** Execute the notebook on the server; confirm `SpatialProp` appears in the niche class-2 board and `summary_table` shows `niche_SpatialProp`.

### Task 1.5: ★ CONCERT comparison checkpoint [SERVER] — gates Phase 2

- [ ] **Step 1:** On Saunders b10, run the benchmark with BOTH `external_models={"SpatialProp": sp_model, "CONCERT": concert_model}`.
- [ ] **Step 2:** Produce `plot_seed_prop` (E-distance boxes) + `plot_delta` (PCC-delta boxes) with both models on the seed board and niche class-2 board (each with its per-prop格2 upper bound + control floor).
- [ ] **Step 3:** Write a one-paragraph finding to `model/benchmark_三模型复现方案_2026-06-25.md` §2.1: do both beat the control floor? which is stronger on seed vs niche? Is the class-2 board shape sane? **Only proceed to Phase 2 once this checkpoint reads sensibly.**

---

## Phase 2 — SpaceTravLR

### Task 2.1: SpaceTravLR input adapter [LOCAL]

**Files:**
- Create: `spbench/adapters/spacetravlr_export.py`
- Test: `tests/test_adapter_spacetravlr.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_spacetravlr.py
import numpy as np, h5py, pytest
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.spacetravlr_export import export_to_spacetravlr_h5

def _d():
    return StandardData(
        X=np.zeros((4, 3)), coords=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float),
        perturbation=np.array(["Gata3", CONTROL, UNLABELED, "Gata3"]),
        cell_type=np.array(["T", "T", "B", "B"]), batch=np.array(["s1", "s1", "s1", "s1"]),
        gene_names=["g1", "g2", "g3"], meta={})

def test_spacetravlr_export_fields(tmp_path):
    d = _d(); X = np.arange(12.).reshape(4, 3)
    info = export_to_spacetravlr_h5(d, X, str(tmp_path / "a.h5ad"), species="mouse")
    assert info["species"] == "mouse"
    with h5py.File(tmp_path / "a.h5ad", "r") as f:
        assert f["X"].shape == (4, 3)
        assert list(np.array(f["obs"]["cell_type"]).astype(str)) == ["T", "T", "B", "B"]
        assert np.allclose(np.array(f["obsm"]["spatial"]), d.coords)
        assert f.attrs["species"] == "mouse"

def test_spacetravlr_export_bad_species(tmp_path):
    with pytest.raises(ValueError):
        export_to_spacetravlr_h5(_d(), np.zeros((4, 3)), str(tmp_path / "a.h5ad"), species="rat")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_adapter_spacetravlr.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'spbench.adapters.spacetravlr_export'`

- [ ] **Step 3: Write minimal implementation**

```python
# spbench/adapters/spacetravlr_export.py
"""StandardData -> SpaceTravLR input. SpaceTravLR (CellOracle/SpaceOracle descendant) needs an
AnnData with RAW counts in X (it builds a base-GRN + CellChat ligand-receptor graph internally in
setup_), obs['cell_type'], obs['batch'], obsm['spatial'], and a species tag that selects the
mouse/human base GRN. SINGLE-CELL resolution only — do NOT export Visium-spot data (Dhainaut)."""
import numpy as np, h5py


def export_to_spacetravlr_h5(data, counts_X, path, species="mouse"):
    counts_X = np.asarray(counts_X, float)
    if counts_X.shape != (data.n_cells, data.n_genes):
        raise ValueError(f"counts_X shape {counts_X.shape} != ({data.n_cells}, {data.n_genes})")
    if species not in ("mouse", "human"):
        raise ValueError(f"species must be 'mouse' or 'human', got {species!r}")
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=counts_X)
        g = f.create_group("obs")
        g.create_dataset("cell_type", data=np.asarray([str(c) for c in data.cell_type], dtype="S"))
        g.create_dataset("batch", data=np.asarray([str(b) for b in data.batch], dtype="S"))
        f.create_group("obsm").create_dataset("spatial", data=np.asarray(data.coords, float))
        f.create_group("var").create_dataset("gene_names",
                        data=np.asarray([str(x) for x in data.gene_names], dtype="S"))
        f.attrs["species"] = species
    return {"n_cells": int(data.n_cells), "species": species}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_adapter_spacetravlr.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add spbench/adapters/spacetravlr_export.py tests/test_adapter_spacetravlr.py
git commit -m "feat: SpaceTravLR input adapter (raw counts + species GRN tag)"
```

### Task 2.2: SpaceTravLR loader [LOCAL]

**Files:**
- Create: `spbench/models/spacetravlr_model.py`
- Test: `tests/test_spacetravlr_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spacetravlr_model.py
import numpy as np, h5py
from spbench.synthetic import make_synthetic
from spbench.graph import build_knn_graph
from spbench.models.spacetravlr_model import SpaceTravLRModel

def _write(path, X, layer):
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=np.zeros_like(X))
        f.create_group("layers").create_dataset(layer, data=np.asarray(X, float))

def test_spacetravlr_predict_niche_and_external(tmp_path):
    from spbench.config import run_benchmark
    data = make_synthetic(0); edges = build_knn_graph(data, k=8)
    P = data.perturbations()[0]; p = tmp_path / f"{P}.h5ad"
    _write(str(p), data.X + 0.5, "predicted_perturbed")
    model = SpaceTravLRModel({P: str(p)})
    niche = model.predict_niche(data, P, edges)
    assert niche.ndim == 2 and niche.shape[1] == data.n_genes
    res = run_benchmark(data, perturbations=[P],
                        external_models={"SpaceTravLR": model}, progress=False)
    assert "SpaceTravLR" in res["compare"][P]["pcc"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spacetravlr_model.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'spbench.models.spacetravlr_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# spbench/models/spacetravlr_model.py
"""SpaceTravLR as an offline end-to-end loader. SpaceTravLR runs in its own uv env (celloracle/commot)
and dumps a per-guide `.h5ad` whose PANEL-ALIGNED perturbed prediction lives in
layers['predicted_perturbed']. The loading + bystander-niche extraction is identical to SpatialProp's,
so we subclass it and only change the model name + default layer."""
from .spatialprop_model import SpatialPropModel


class SpaceTravLRModel(SpatialPropModel):
    name = "spacetravlr"

    def __init__(self, prediction_paths, layer="predicted_perturbed"):
        super().__init__(prediction_paths, layer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_spacetravlr_model.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add spbench/models/spacetravlr_model.py tests/test_spacetravlr_model.py
git commit -m "feat: SpaceTravLR loader (dump-niche, predicted_perturbed layer)"
```

### Task 2.3: SpaceTravLR runner — pure helpers [LOCAL]

**Files:**
- Create: `scripts/spacetravlr/run_spacetravlr.py`
- Test: `tests/test_runner_spacetravlr.py`

The `main()` (setup_ → fit → setup_perturbations → perturb, bypassing `spawn_worker`) is `# pragma: no cover` (needs the spacetravlr env + GPU). Only the pure eligibility helper is unit-tested here.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner_spacetravlr.py
import os, importlib.util
_R = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                  "scripts", "spacetravlr", "run_spacetravlr.py")

def _load():
    spec = importlib.util.spec_from_file_location("run_spacetravlr", _R)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_is_perturbable_tf_or_lr():
    m = _load()
    assert m.is_perturbable("Gata3", ["Gata3", "Il2"]) is True
    assert m.is_perturbable("Xyz", ["Gata3", "Il2"]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner_spacetravlr.py -q`
Expected: FAIL with `FileNotFoundError` (the script does not exist yet)

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/spacetravlr/run_spacetravlr.py
"""Offline SpaceTravLR runner — dedicated uv env (celloracle/commot/torch + GPU). We BYPASS
SpaceShip.spawn_worker (which self-submits a Slurm job) and drive the model directly inside our own
srun allocation: setup_ -> fit -> setup_perturbations -> perturb. KO = perturb(target, gene_expr=0).
The perturbed matrix is aligned onto the exported panel via align_prediction_to_panel (genes
SpaceTravLR doesn't model are left at the unperturbed input) and dumped to
layers['predicted_perturbed']. Off-panel / off-GRN targets are skipped (no-op, option 2). The pure
helper below is unit-tested in the venv; main() needs the real env."""
import argparse
import numpy as np


def is_perturbable(target, perturbable):
    """SpaceTravLR can only perturb a TF (base GRN) or ligand/receptor (CellChat). `perturbable` is
    that allowed-gene set. Returns True iff `target` is injectable; else the runner skips it."""
    return str(target) in set(map(str, perturbable))


def main():                                          # pragma: no cover (needs spacetravlr env + GPU)
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)         # from export_to_spacetravlr_h5
    ap.add_argument("--pert", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    import h5py                                       # lazy
    import scanpy as sc
    from SpaceTravLR.spaceship import SpaceShip
    from spbench.dump_align import align_prediction_to_panel

    adata = sc.read_h5ad(args.h5ad)
    species = adata.uns.get("species", "mouse") if hasattr(adata, "uns") else "mouse"
    panel = list(adata.var["gene_names"].astype(str))
    ship = SpaceShip(name="bench").setup_(adata, run_commot=True)          # GRN + L-R preprocessing
    # perturbable set = base-GRN TFs ∪ CellChat ligands/receptors for this species
    perturbable = set(SpaceShip.load_base_GRN(species).iloc[:, 0].astype(str))
    if not is_perturbable(args.pert, perturbable):
        raise SystemExit(f"target {args.pert!r} is not a TF/ligand/receptor SpaceTravLR can perturb")
    ship.fit(epochs=args.epochs)                                          # = run_spacetravlr (no sbatch)
    ship.setup_perturbations(adata)
    pred = np.asarray(ship.perturb(target=args.pert, gene_expr=0, propagation=4), float)
    pred_genes = list(ship.factory.genes) if hasattr(ship, "factory") else panel
    fallback = np.asarray(adata.X, float)
    aligned = align_prediction_to_panel(pred, pred_genes, panel, fallback)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("X", data=np.zeros_like(aligned))
        f.create_group("layers").create_dataset("predicted_perturbed", data=aligned)
    print(f"wrote {args.out}: predicted_perturbed {aligned.shape}")


if __name__ == "__main__":                           # pragma: no cover
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runner_spacetravlr.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/spacetravlr/run_spacetravlr.py tests/test_runner_spacetravlr.py
git commit -m "feat: SpaceTravLR runner (bypass spawn_worker; panel-aligned dump)"
```

### Task 2.4: SpaceTravLR env recipe [LOCAL]

**Files:**
- Create: `scripts/spacetravlr/ENV_SETUP.md`

- [ ] **Step 1: Write the recipe doc** (no test — documentation)

```markdown
# SpaceTravLR offline env (server)

SpaceTravLR self-submits Slurm via `spawn_worker`; we DO NOT use it — the runner calls
`setup_ → fit → setup_perturbations → perturb` directly inside an `srun` allocation.

## Build (uv, on the server)
    cd ~ && git clone https://github.com/jishnu-lab/SpaceTravLR && cd SpaceTravLR
    git checkout release
    uv venv ~/.venvs/spacetravlr && source ~/.venvs/spacetravlr/bin/activate
    uv sync                              # installs SpaceTravLR + celloracle + commot + torch
    # GPU: node03 (RTX 3090). Verify: python -c "import torch; print(torch.cuda.is_available())"

## Smoke
    python -c "from SpaceTravLR.spaceship import SpaceShip; print(SpaceShip.load_base_GRN('mouse').shape)"

## Notes
- Single-cell datasets only (Saunders/Shen/Binan/Cheng); Dhainaut (Visium spot) is out of scope.
- `setup_(run_commot=True)` runs CellOracle (TF GRN) + COMMOT (ligand-receptor) — slow; cache per dataset.
- Coverage: only guides in `load_base_GRN(species)` (TF) ∪ CellChat L-R are injectable (Task 0.1).
```

- [ ] **Step 2: Commit**

```bash
git add scripts/spacetravlr/ENV_SETUP.md
git commit -m "docs: SpaceTravLR offline env recipe (uv, bypass spawn_worker)"
```

### Task 2.5: Build env + run SpaceTravLR on a single-cell dataset [SERVER]

- [ ] **Step 1:** Build `~/.venvs/spacetravlr` per `ENV_SETUP.md`; smoke-check `load_base_GRN`.
- [ ] **Step 2:** Export Saunders b10 via `export_to_spacetravlr_h5(d, build_counts_X(d), out, species="mouse")` (concert env).
- [ ] **Step 3:** Run the coverage helper to get the injectable guide list (TF∪L-R); for each, run `scripts/spacetravlr/run_spacetravlr.py` → `~/spacetravlr_dumps_Saunders_b10/{P}.h5ad`.
- [ ] **Step 4:** In the notebook, construct `SpaceTravLRModel({P: ...})`, add to `external_models`; confirm it scores on the seed + niche class-2 boards.

---

## Phase 3 — Celcomen

### Task 3.1: Celcomen input adapter [LOCAL]

**Files:**
- Create: `spbench/adapters/celcomen_export.py`
- Test: `tests/test_adapter_celcomen.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_adapter_celcomen.py
import numpy as np, h5py, pytest
from spbench.data import StandardData, CONTROL, UNLABELED
from spbench.adapters.celcomen_export import export_to_celcomen_h5

def _d(counts):
    return StandardData(
        X=np.zeros((4, 3)), coords=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float),
        perturbation=np.array(["GeneA", CONTROL, UNLABELED, "GeneA"]),
        cell_type=np.array(["T", "T", "B", "B"]), batch=np.array(["s1", "s1", "s1", "s1"]),
        gene_names=["g1", "g2", "g3"], meta={})

def test_celcomen_export_fields(tmp_path):
    counts = np.array([[2, 4, 0], [0, 6, 2], [4, 0, 8], [2, 2, 2]], float)
    d = _d(counts)
    export_to_celcomen_h5(d, counts, str(tmp_path / "a.h5ad"))
    with h5py.File(tmp_path / "a.h5ad", "r") as f:
        assert np.allclose(np.array(f["X"]), counts)
        assert list(np.array(f["obs"]["cell_type"]).astype(str)) == ["T", "T", "B", "B"]
        assert np.allclose(np.array(f["obsm"]["spatial"]), d.coords)

def test_celcomen_export_rejects_non_integer(tmp_path):
    d = _d(None)
    with pytest.raises(ValueError):
        export_to_celcomen_h5(d, np.full((4, 3), 0.5), str(tmp_path / "a.h5ad"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_adapter_celcomen.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'spbench.adapters.celcomen_export'`

- [ ] **Step 3: Write minimal implementation**

```python
# spbench/adapters/celcomen_export.py
"""StandardData -> Celcomen input. Celcomen (CCE inference + Simcomen counterfactual) needs RAW
integer counts (no normalization), obs['cell_type'], obs['batch'], obsm['spatial'] (it builds the
k-hop spatial graph internally), and gene names. KO is applied in Simcomen via set_sphex on the
in-panel guide gene's column."""
import numpy as np, h5py


def export_to_celcomen_h5(data, counts_X, path):
    counts_X = np.asarray(counts_X, float)
    if counts_X.shape != (data.n_cells, data.n_genes):
        raise ValueError(f"counts_X shape {counts_X.shape} != ({data.n_cells}, {data.n_genes})")
    if not np.allclose(counts_X, np.round(counts_X), atol=1e-6):
        raise ValueError("Celcomen requires RAW INTEGER counts; pass the raw count layer.")
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=counts_X)
        g = f.create_group("obs")
        g.create_dataset("cell_type", data=np.asarray([str(c) for c in data.cell_type], dtype="S"))
        g.create_dataset("batch", data=np.asarray([str(b) for b in data.batch], dtype="S"))
        f.create_group("obsm").create_dataset("spatial", data=np.asarray(data.coords, float))
        f.create_group("var").create_dataset("gene_names",
                        data=np.asarray([str(x) for x in data.gene_names], dtype="S"))
    return {"n_cells": int(data.n_cells)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_adapter_celcomen.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add spbench/adapters/celcomen_export.py tests/test_adapter_celcomen.py
git commit -m "feat: Celcomen input adapter (raw integer counts + spatial)"
```

### Task 3.2: Celcomen loader [LOCAL]

**Files:**
- Create: `spbench/models/celcomen_model.py`
- Test: `tests/test_celcomen_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_celcomen_model.py
import numpy as np, h5py
from spbench.synthetic import make_synthetic
from spbench.graph import build_knn_graph
from spbench.models.celcomen_model import CelcomenModel

def _write(path, X, layer):
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=np.zeros_like(X))
        f.create_group("layers").create_dataset(layer, data=np.asarray(X, float))

def test_celcomen_predict_niche_and_external(tmp_path):
    from spbench.config import run_benchmark
    data = make_synthetic(0); edges = build_knn_graph(data, k=8)
    P = data.perturbations()[0]; p = tmp_path / f"{P}.h5ad"
    _write(str(p), data.X + 0.5, "counterfactual")
    model = CelcomenModel({P: str(p)})
    niche = model.predict_niche(data, P, edges)
    assert niche.ndim == 2 and niche.shape[1] == data.n_genes
    res = run_benchmark(data, perturbations=[P],
                        external_models={"Celcomen": model}, progress=False)
    assert "Celcomen" in res["compare"][P]["pcc"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_celcomen_model.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'spbench.models.celcomen_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# spbench/models/celcomen_model.py
"""Celcomen as an offline end-to-end loader. Celcomen runs in its own python=3.9 env (CCE training +
Simcomen counterfactual generation) and dumps a per-guide `.h5ad` whose counterfactual expression
lives in layers['counterfactual']. Loading + bystander-niche extraction is identical to SpatialProp's
so we subclass it and only change the model name + default layer."""
from .spatialprop_model import SpatialPropModel


class CelcomenModel(SpatialPropModel):
    name = "celcomen"

    def __init__(self, prediction_paths, layer="counterfactual"):
        super().__init__(prediction_paths, layer)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_celcomen_model.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add spbench/models/celcomen_model.py tests/test_celcomen_model.py
git commit -m "feat: Celcomen loader (dump-niche, counterfactual layer)"
```

### Task 3.3: Celcomen runner — pure helper [LOCAL]

**Files:**
- Create: `scripts/celcomen/run_celcomen.py`
- Test: `tests/test_runner_celcomen.py`

`main()` (CCE train → Simcomen `set_sphex` KO → generate) is `# pragma: no cover`. Only the in-panel KO-index helper is unit-tested.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner_celcomen.py
import os, importlib.util
_R = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                  "scripts", "celcomen", "run_celcomen.py")

def _load():
    spec = importlib.util.spec_from_file_location("run_celcomen", _R)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_ko_gene_index_in_panel():
    m = _load()
    assert m.ko_gene_index("g2", ["g1", "g2", "g3"]) == 1

def test_ko_gene_index_off_panel_none():
    m = _load()
    assert m.ko_gene_index("gX", ["g1", "g2"]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner_celcomen.py -q`
Expected: FAIL with `FileNotFoundError` (the script does not exist yet)

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/celcomen/run_celcomen.py
"""Offline Celcomen runner — dedicated python=3.9 env (torch/PyG). Two stages: (1) CCE learns the
gene-to-gene + intracellular regulation matrices; (2) Simcomen sets those matrices and applies the KO
via set_sphex (zero the guide gene's column), then generates the counterfactual spatial expression,
dumped to layers['counterfactual']. Off-panel guides have no column to zero -> skipped (no-op,
option 2). The pure helper below is unit-tested in the venv; main() needs the celcomen env."""
import argparse
import numpy as np


def ko_gene_index(guide_gene, genes):
    """Column index of the KO guide gene for Simcomen set_sphex zeroing. Returns None when the guide
    is OFF-PANEL (Celcomen can only KO an in-panel gene) — the runner then skips it."""
    genes = list(map(str, genes))
    return genes.index(str(guide_gene)) if str(guide_gene) in genes else None


def main():                                          # pragma: no cover (needs celcomen env)
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)         # from export_to_celcomen_h5
    ap.add_argument("--pert", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import h5py                                       # lazy
    import scanpy as sc
    from celcomen.models.celcomen import Celcomen
    from celcomen.models.simcomen import Simcomen

    adata = sc.read_h5ad(args.h5ad)
    genes = list(adata.var["gene_names"].astype(str))
    j = ko_gene_index(args.pert, genes)
    if j is None:
        raise SystemExit(f"guide {args.pert!r} is off-panel; Celcomen cannot KO it")
    # (1) CCE: learn g2g + intracellular matrices  (2) Simcomen: set matrices, KO gene j, generate.
    # Exact CCE/Simcomen call signatures follow Tutorial_Celcomen_on_Xenium.ipynb / spatial_KO tutorial.
    cce = Celcomen(input_dim=len(genes), output_dim=len(genes), n_neighbors=6)
    cce.fit(adata)                                   # learns g2g, g2g_intra (see tutorial)
    sim = Simcomen(input_dim=len(genes), output_dim=len(genes), n_neighbors=6)
    sim.set_g2g(cce.get_g2g()); sim.set_g2g_intra(cce.get_g2g_intra())
    sphex = sim.x_to_sphex(np.asarray(adata.X, float))
    sphex[:, j] = 0.0                                # knock out guide gene
    sim.set_sphex(sphex)
    cf = np.asarray(sim.sphex_to_x(sim.generate()), float)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("X", data=np.zeros_like(cf))
        f.create_group("layers").create_dataset("counterfactual", data=cf)
    print(f"wrote {args.out}: counterfactual {cf.shape}")


if __name__ == "__main__":                           # pragma: no cover
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_runner_celcomen.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/celcomen/run_celcomen.py tests/test_runner_celcomen.py
git commit -m "feat: Celcomen runner (CCE+Simcomen KO; counterfactual dump)"
```

### Task 3.4: Celcomen env recipe [LOCAL]

**Files:**
- Create: `scripts/celcomen/ENV_SETUP.md`

- [ ] **Step 1: Write the recipe doc** (no test)

```markdown
# Celcomen offline env (server)

## Build
    conda create -n celcomen python=3.9 -y && conda activate celcomen
    pip install git+https://github.com/stathismegas/celcomen
    # torch + PyG per the repo's pyproject; GPU node03.

## Smoke
    python -c "from celcomen.models.simcomen import Simcomen; print('ok')"

## Notes
- RAW integer counts, NO normalization (the adapter enforces this).
- KO = set_sphex zero on the in-panel guide column; off-panel guides are skipped (option 2).
- Confirm the exact CCE.fit / Simcomen generate signatures against
  Tutorial_Celcomen_on_Xenium.ipynb and the spatial_KO tutorial before the first real run;
  adjust `scripts/celcomen/run_celcomen.py::main` if the API differs.
```

- [ ] **Step 2: Commit**

```bash
git add scripts/celcomen/ENV_SETUP.md
git commit -m "docs: Celcomen offline env recipe (python=3.9)"
```

### Task 3.5: Build env + run Celcomen [SERVER]

- [ ] **Step 1:** Build `~/.conda/envs/celcomen` per `ENV_SETUP.md`; smoke-check `Simcomen` import; verify the CCE/Simcomen API against the tutorial and patch `run_celcomen.py::main` if needed (commit + re-sync).
- [ ] **Step 2:** Export Saunders b10 via `export_to_celcomen_h5` (concert env) using the raw count layer.
- [ ] **Step 3:** Run the in-panel guides (Task 0.1 coverage) through `scripts/celcomen/run_celcomen.py` → `~/celcomen_dumps_Saunders_b10/{P}.h5ad`.
- [ ] **Step 4:** Construct `CelcomenModel({P: ...})` in the notebook, add to `external_models`; confirm seed + niche class-2 scoring.

---

## Final verification [LOCAL]

- [ ] Run the full local suite — everything green (one known pre-existing flaky test may need a re-run):

Run: `python -m pytest -q`
Expected: all pass (re-run once if the single known-flaky test trips).

- [ ] Confirm the three new models all expose `predict_niche` returning `(m, n_genes)` and score via `run_benchmark(external_models=...)` (covered by Tasks 2.2 / 3.2 tests).

---

## Self-review notes (author)

- **Spec coverage:** SpatialProp finish (Phase 1 + §2.1 CONCERT checkpoint), SpaceTravLR bypass-spawn_worker (Task 2.3 `main`), Celcomen CCE+Simcomen (Task 3.3 `main`), offline-dump template per model (adapter/runner/loader/env/tests), off-panel option 2 (coverage helper 0.1 + per-runner skip), panel alignment for subset-gene models (0.2), coverage quantification (0.1 + [SERVER] runs) — all mapped.
- **Open items deliberately left to the first server run (flagged in-task, not placeholders):** exact SpatialProp import name (1.1), SpaceTravLR `ship.factory.genes` attribute name (2.3 — falls back to panel), Celcomen CCE/Simcomen exact call signatures (3.3 — pinned to the tutorial in 3.5). These need the real packages to confirm and cannot be unit-tested locally.
- **Type/name consistency:** loaders subclass `SpatialPropModel`; dump layers are `predicted_tempered` (SpatialProp) / `predicted_perturbed` (SpaceTravLR) / `counterfactual` (Celcomen), each matched between its runner dump and loader default.
