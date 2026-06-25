# Retire E-distance → PCC-delta primary + per-seed-model niche (incl. scGEN in lognorm)

> Branch `feat/retire-e-pcc-delta` (worktree `/Users/cz/Documents/ZengLab/model/sppb-wt-edist`, off `main`).
> Worktree test cmd: `cd /Users/cz/Documents/ZengLab/model/sppb-wt-edist && .venv/bin/python -m pytest -q`
> (its venv: spbench editable → worktree; heavy deps shared from the orig `.venv` via `_origdeps.pth`).
> Isolation: the orig checkout is on another task's branch (`codex/...`) — do NOT touch it. Server
> re-run later must use a SEPARATE server dir (e.g. `~/sp-perturb-benchmark-edist`), not the shared one.

## Decisions (settled with user)
- **E-distance fully retired** from the benchmark headline (base_df / capability matrix / plots). Energy
  is a distribution-vs-distribution metric; our predictions are mean-field (one seed mean → propagated),
  so E collapses. Keep `metrics/energy.py` + the `e`/`e_samples` fields in `compare` (cheap, internal),
  but nothing E-based is shown/ranked.
- **Primary metric = PCC-delta (mean-shift direction) + magnitude** (`‖Δpred‖/‖Δtrue‖`), for both seed
  and niche. `compare_to_baseline.pcc` (niche) and `evaluate_seed.pcc_delta` (seed) already exist.
- **Every seed model gets seed + niche** (niche via the shared baseline Gaussian prop). Capability
  matrix rows become per-seed-model: TrivialSeed (seed+niche), scGEN (seed+niche).
- **scGEN's niche is computed in its log-norm space** (option ③): co-space the whole niche path into
  `eval_X` when it's the lognorm matrix, then `compare_to_baseline` on scGEN's grid.
- **Cross-space comparability = normalized gap**: each model's PCC-delta normalized between its OWN
  null (≈0) and its OWN GT-seed upper (per dataset, same space). Raw PCC shown too, with a "spaces
  differ" caveat. This extends `aggregate.normalized_score` to PCC.

## Parts

### A. Magnitude metric + normalized-gap helper (library, TDD)
- `spbench/compare.py`: add a `mag` per method to `compare_to_baseline` and a `mag` to `evaluate_seed`
  = `‖Δpred‖ / ‖Δtrue‖` where `Δpred = mean(pred)−mean(ref)`, `Δtrue = mean(obs)−mean(ref)` (in eval_X
  space, same as pcc). 1.0 = right magnitude; <1 under-, >1 over-shoot.
- `spbench/aggregate.py`: add `normalized_pcc(pcc, pcc_null, pcc_upper)` → `(pcc − pcc_null)/(pcc_upper
  − pcc_null)` clipped [0,1] (null≈0, upper=GT-seed). Tests: null→0, upper→1, mid→0.5, degenerate→nan.

### B. fill_2x2 niche co-space (enables scGEN niche) (library, TDD)
- `spbench/harness.py::fill_2x2`: when `isinstance(eval_X, np.ndarray)`, build the NICHE in that space:
  observed/reference niche = `eval_X[pert_nb]`/`eval_X[ref_nb]`, `X_ref` from eval_X per-cell-type
  control means, residuals from eval_X, propagate on the eval_X `X_ref`. (Today only the seed is
  co-spaced; the niche stays data.X → a lognorm seed propagated on data.X is space-mixed.)
  - Need the bystander indices: have `propagation_gt` also return `pert_nb`/`ref_nb` (or recompute in
    fill_2x2 via `_bystanders`). Add a `_control_residuals_X(Xexpr, data)` / `_control_reference_aggregate_X`
    that take an explicit expression matrix.
- Test: a synthetic where eval_X = a known transform of data.X → scGEN-style seed_model produces a
  niche whose `compare_to_baseline` pcc is finite and the niche cells live in eval_X space.

### C. run_benchmark: per-seed-model niche
- Let the notebook/`run_benchmark` score EACH seed model's niche via `compare_to_baseline` on its
  fill_2x2 grid (TrivialSeed in data.X; scGEN in lognorm via B). Expose `res['compare'][p]` per model
  or a `res['models'][name]` map. (Minimal: keep run_benchmark as-is for TrivialSeed; in the notebook
  §6, after the scGEN fill_2x2, ALSO call `compare_to_baseline(niches)` → scgen niche pcc/mag.)

### D. Notebook rewire (the pipeline face)
- base_df: drop `*_E` headline columns; primary = `seed_pcc`, `niche_pcc` (+ `*_mag`); keep null/upper
  PCC anchors for the normalized gap. scGEN: add `scgen_seed_pcc`, `scgen_niche_pcc`.
- `_capability_matrix`: rows per seed model (TrivialSeed: seed+niche; scGEN: seed+niche; null/oracle as
  PCC anchors), values = PCC-delta AND normalized-gap. Drop the E heatmap.
- Headline plot = `plot_delta` (PCC-delta seed|niche). Remove `plot_seed_prop` (E) from the notebook.
- Prose/§ numbering updated; regenerate `notebooks/run_benchmark.ipynb`.

### E. Verify + re-run
- Full worktree suite green. Then sync to a SEPARATE server dir + re-run on Saunders (TrivialSeed +
  scGEN with dumps if available) → confirm scGEN now has a niche PCC + the normalized-gap table.

## Status
- [x] A  - [x] B  - [x] C  - [x] D  - [x] E

## Part E results (Saunders Batch_10_Slice_0, 14 EVAL guides, server `~/sp-perturb-benchmark-edist`)
scGEN dumps generated (14/14 `{P}_seed.h5ad`, scgen env) + notebook re-run (nbconvert, exit 0).
**Every seed model now has BOTH a seed and a niche PCC-delta** — capability matrix:

| model | seed_pcc | niche_pcc | seed_norm | niche_norm |
|---|---|---|---|---|
| TrivialSeed | 0.275 | −0.065 | 0.285 | 0.998 |
| scGEN | 0.379 | −0.107 | 0.379 | 1.000 |

- ✅ Core deliverable met: `scgen_niche_pcc` exists (per-guide + mean), computed in scGEN's log-norm
  space via the Part-B co-spacing; cross-space normalized gap computed.
- ✅ Sanity: scGEN seed > TrivialSeed seed (0.379 vs 0.275; scGEN higher on 8/14 guides) — a
  conditional model beats the cell-type-mean seed.
- ⚠️ **Finding (not a bug): niche prediction is weak on this slice** — niche_pcc ~ 0/slightly
  negative for BOTH models, i.e. the Gaussian-propagated niche shift does not correlate with the
  observed bystander shift. The GT-seed niche upper (`GT+base`) is itself ~0, so `niche_norm` is
  NaN for most guides (my no-headroom guard) and the ~1.0 mean is built from only 2–3 finite values
  → **do not read niche_norm ~1.0 as "niche solved"**. The honest read is raw niche_pcc < 0.
- Artifacts on server: `~/sp-perturb-benchmark-edist/notebooks/run_benchmark.executed.ipynb`,
  `run_benchmark_results.png`, `delta_methods.png`; dumps `~/scgen_dumps_edist/Saunders_b10/`.
- ⚠️ Slice gotcha: notebook default `SLICE_DIR`=dir + `MAX_FILES=1` loads the ALPHABETICALLY-FIRST
  slice (`Batch_0_Slice_0`, sparse: 55 controls), NOT the `Batch_10_Slice_0` the EVAL list assumes.
  Part E pinned it via a single-file symlink dir (`~/saunders_b10_slice`). Documented in the notebook.
