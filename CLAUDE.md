# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`spbench` — a benchmark for **spatial single-cell perturbation prediction**: does a model correctly
predict how a CRISPR knockout reshapes gene expression in intact tissue, both in the perturbed cell
itself (**seed**) and in its spatial neighbourhood (**niche**)? Read `README.md` for the method
rationale (no paired before/after → distribution-level scoring; control-based reference niche → no
leakage; the seed × propagation 2×2). This file covers what the README doesn't: how to run things,
the code's big picture, and where the data + server live.

## Commands

The package is pure-Python and runs **locally on a built-in synthetic generator** — no real data or
GPU needed for development or tests. Real datasets live only on the lab server (see below).

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e .   # dev install
python -m pytest -q                                                     # full suite (testpaths=tests)
python -m pytest tests/test_compare_samples.py::test_name -q            # single test
python notebooks/build_notebook.py                                      # regenerate notebooks/run_benchmark.ipynb
```

- Tests run entirely on synthetic data; `tests/conftest.py` provides the `synth` fixture
  (`make_synthetic(seed=0)`). When adding a feature, drive it from `spbench.synthetic` so the test
  needs no real data.
- Quick end-to-end on synthetic:
  `from spbench.config import run_benchmark; from spbench.synthetic import make_synthetic;
   run_benchmark(make_synthetic(0), perturbations=["P0"], gcn_kwargs={"hidden":16,"epochs":5}, progress=False)`
- Config-driven run on a real dataset (server only): `spbench.config.run_from_yaml("configs/saunders_mvp.yaml")`.
- There is **one known pre-existing flaky test** (~1/8 runs, unseeded randomness in an older test); a
  re-run is green. Not introduced by recent work.

## Architecture (the pipeline, end to end)

Top-level orchestrator is `spbench/config.py::run_benchmark(data, ...)`. The data flow:

1. **Input adapter** (`spbench/adapters/*`) normalises any dataset into one `StandardData`
   (`spbench/data.py`: `X`, `coords`, `perturbation` [gene | `'control'` | `'none'`], `cell_type`,
   `batch`, `gene_names`). Adapters self-register in `adapters/__init__.py::_REGISTRY` (`get_adapter`).
   Key property: `data.control_pool` = the reference cell set — `is_control` cells, else falls back to
   `'none'` cells (datasets with no NTC), else all cells.
2. **Reference + graph**: `graph.build_knn_graph` (within-batch kNN); `reference_aggregate`
   builds the **sample-level aggregate control** — each perturbed cell's reference is ALL same-cell-type
   control cells (NO feature-space nearest-neighbour matching; that retired matcher was the old
   leakage source). `harness._control_reference_aggregate` / `_control_residuals` derive the per-cell
   reference matrix and the per-cell-type control-residual pools.
3. **Per perturbation**: `harness.fill_2x2(data, p, edges, seed_model, baseline_prop, learned_prop, ...)`
   fills the 2×2 of {GT seed, model seed} × {baseline (Gaussian) prop, learned (GCN) prop}, returns the
   predicted niche clouds for all four cells plus `_niches` (observed / reference / per-cell seed arrays).
   `propagation_gt` supplies the observed perturbed niche (bystander neighbours of perturbed centers)
   vs the reference niche (bystanders of control centers).
4. **Scoring** (`spbench/compare.py`):
   - `compare_to_baseline(niches, ...)` → **niche** scores: matched-n energy distance per 2×2 cell +
     `null` (no-effect) + `oracle` ceiling, plus a niche **PCC-delta** per method. `extra={name: arr}`
     scores an external/end-to-end model's niche on the same footing.
   - `evaluate_seed(niches, ...)` → **seed** scores: PCC-delta (direction) + MSE + per-repeat energy.
   - Metric primitives are in `spbench/metrics/` (registry: `get_metric`); only `energy`, `pcc_delta`,
     `mse` are used by live scoring. `metrics/energy.py` is the Székely energy distance.
5. **Attribution / null / plots**: `judge.attribute`/`leakage_pass`; `permutation.permutation_null`
   (optional `n_perm`, empirical p per perturbation); `plotting.plot_seed_prop` (E-distance boxes) and
   `plotting.plot_delta` (PCC-delta boxes); `aggregate.cross_dataset_rank` (normalize-then-rank across
   datasets, since absolute E is not comparable across platforms).

Models implement small ABCs in `spbench/models/base.py`: `SeedModel.predict_seed`,
`PropModel.propagate`, `EndToEndModel.predict_niche`. Seed models self-register (`get_model`) except
loaders that need constructor args (`ScgenSeedModel`, `ConcertModel` — constructed with dump paths).

### Two invariants that are easy to break
- **eval_X (scoring space).** PCC-delta is **not** cross-space robust, so prediction, observed, and the
  delta baseline must be in ONE space. scGEN is scored in its log-norm space — `fill_2x2` slices the
  seed cells into the `build_lognorm_X` matrix; baselines stay in `data.X`. Don't compare raw scores
  across spaces.
- **seed vs niche are distributional but predictions are often mean-field.** Energy distance is
  distribution-vs-distribution; a collapsed point prediction (mean-field seed) is structurally
  penalised regardless of whether its mean is right. The current direction is to make **PCC-delta
  (mean-shift direction + magnitude) the primary metric** and demote E-distance to genuinely
  distributional predictors. See `model/明早问_2026-06-25.md` for the live design decisions.

## Datasets (real data — lab server only, read-only under `/home/yiru/`)

| Adapter key | Dataset | Path | Notes |
|---|---|---|---|
| `saunders` | Saunders 2025 MERFISH liver (core) | `/home/yiru/database/spatial_perturbed_processed/CRISPR_based/Saunders_2025_40513557/` (`.h5mu`) | The only processed version that's clean; single-cell, 9 cell types |
| `dhainaut` | Dhainaut 2022 Perturb-map (Visium spot) | `/home/yiru/database/spatial_perturbation_collection/Perturb_Map/GSE193460_RAW/` (spaceranger) | Use RAW whole-transcriptome, NOT yiru's 2000-panel `.h5mu` (missing T-cell / target markers); control = KP base |
| `binan_tumors` | Binan Perturb-FISH tumors | `/home/yiru/database/spatial_perturbation_collection/Perturb_FISH/finaltables_downloads/tumors/processed/finaltables/` | all-cells; named guides from `tumorpooledperturbations`; immune-neighbour annotation in meta |
| `cheng` | Cheng Perturb-RAEFISH A549 | `/home/yiru/database/spatial_perturbation_collection/REAFISH_data/Datasets/A549/Perturb_RAEFISH/` (`.mat` + codebooks) | single cell line (degenerate mode); `Codebook_RaeFISH.Target` maps Top1ID→KO gene |
| `shen` | Shen Spatial Perturb-seq brain | `/home/yiru/database/spatial_perturbed_processed/CRISPR_based/Shen_2024_preprint/` (`.h5mu`) | whole-transcriptome, no NTC (uses `control_pool` fallback); `mSafe`→control, `Non-perturbed`→none |

Per-dataset adaptation details + gotchas: `model/空间扰动benchmark_数据集适配记录_2026-06-24.md` and
`model/Dhainaut数据集_问题与处理_2026-06-24.md` (these design docs live in the parent `model/`
workspace, not the repo). Each adapter has a `runs/verify_<dataset>.py` smoke script that loads the
real data on the server.

## Running on the lab server (`zenglab`)

The repo is **not pip-installed on the server** — source is rsync'd to `~/sp-perturb-benchmark`. GitHub
is blocked on the cluster, so sync with `git archive HEAD | ssh zenglab 'tar -x -C ~/sp-perturb-benchmark'`
(+ `rsync -aqz ./.git/ zenglab:~/sp-perturb-benchmark/.git/` when history is needed), never `git pull`.

Compute is **Slurm-only** — do NOT run notebooks / h5py / heavy Python on the login node. Discover the
user's running JupyterLab allocation (`squeue -u $(whoami)`) and attach with
`srun --jobid=<jobid> --overlap`. Env essentials for any run:

```bash
srun --jobid=<jobid> --overlap bash -lc '
  export LD_LIBRARY_PATH=$HOME/.conda/envs/concert/lib:$LD_LIBRARY_PATH   # h5py needs this GLIBC
  export PYTHONPATH=$HOME/sp-perturb-benchmark:$PYTHONPATH                # source is not installed
  $HOME/.conda/envs/concert/bin/python ...'
```

- GPU: `-w node03` (RTX 3090, 24 GB) and do NOT pass `--gres`; `node05` has no GPU. Check VRAM before
  launching on a shared card.
- The `concert` conda env is the main env. **scGEN runs in a SEPARATE offline env** (scvi-tools 0.20
  pins conflict with the main env) — recipe in `scripts/scgen/ENV_SETUP.md`; the offline runner is
  `scripts/scgen/run_scgen.py` (exports `StandardData` → log-norm AnnData, trains one scGEN per
  perturbation, dumps `{P}_seed.h5ad`). `ScgenSeedModel` / `ConcertModel` are loaders that read those
  offline dumps.
- The `ssh-notebook-debug-loop` and `zenglab-server` skills encode the full remote workflow.

## Design docs & history

- Implementation plans: `docs/superpowers/plans/` (`00-roadmap.md` is the index; Plans 1–8 are done).
- The evaluation framework (two model classes, three boards, per-prop upper bounds, seed/niche naming)
  and the overall design live in the parent `model/` workspace:
  `benchmark_评测框架_两类model_2026-06-25.md`, `benchmark_总设计_实现参考_2026-06-24.md`.
- Open design decisions in flight: `model/明早问_2026-06-25.md` (metric → PCC-delta, per-cell-type /
  niche-conditional stratified scoring, scGEN-niche space handling, unused-metric cleanup).
