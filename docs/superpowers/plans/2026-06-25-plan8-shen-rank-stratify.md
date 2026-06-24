# Plan 8 — Shen adapter + cross-dataset rank + MC stratification (DONE)

Three parts, all landed on branch `feat/plan8-cross-dataset-rank` (TDD, full suite green, Shen server-verified):

## (a) ShenAdapter — `spbench/adapters/shen.py`
Stereo-seq brain, single-cell h5mu, whole-transcriptome ~21590 genes. `mod/rna/X` (sparse counts), `obs/perturbation`, `obs/x`+`obs/y` coords, `var/_index` genes. No cell_type column → constant `'brain'` (Plan-2 degenerate; n small ~165 perturbed cells → report n). `_map_shen_perturbation`: gene → KO; `Doublet`/`Non-perturbed` → `'none'` (Plan-2 control pool); `msafe`/safe/NTC → `'control'`. Registered `"shen"`. Pure-function mapping test (`tests/test_adapter_shen.py`); h5mu I/O server-verified.
- **Server verify (node03)**: 20628 cells × 21590 genes, 16 KO, 5 control (msafe), single `brain`; KO=clu seed pcc 0.930, perm p 0.048, niche scored. ✓
- **Flagged for user** (`model/明早问_2026-06-25.md` Q3): `Non-perturbed`→none-vs-control, and whether `msafe` is really a control guide.
- **Bug fixed en route**: h5mu stores counts as `uint32`; `torch.tensor` rejects uint32 → cast `X` to float in the adapter + defensive `np.asarray(..., float32)` in `models/gcn_prop.py` (benefits all adapters).

## (b) Cross-dataset rank — `spbench/aggregate.py`
Absolute E is not comparable across datasets (Visium whole-transcriptome ~400 vs MERFISH aggregated <1 — the CONCERT-magnitude finding). So: `normalized_score(e, e_null, e_oracle)` → [0,1] within a dataset (0=null floor, 1=oracle ceiling); `cross_dataset_rank(per_dataset)` ranks methods per dataset (rank 1 = lowest e) and aggregates ranks + normalized scores across datasets; `rank_from_results({dataset: run_benchmark res})` convenience. Tests in `tests/test_aggregate.py` (incl. a Visium-vs-MERFISH magnitude-incomparability case where ranks still align).

## (c) MC-spatial quadrant × dimension stratification — ALREADY EXISTS
`spbench/mc_spatial_join.py::join_quadrants` + `spbench/mc_spatial_report.py::stratified_report` (quadrant × dimension, Inert quadrant as negative control) were built earlier (`tests/test_mc_spatial_join.py`, `tests/test_mc_spatial_report.py`) and are wired into the notebook's `HAVE_MC` path. Plan 8's "reuse existing" → satisfied, no new code.
