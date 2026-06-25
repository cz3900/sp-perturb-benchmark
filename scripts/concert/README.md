# Evaluating CONCERT in this benchmark (Phase B)

CONCERT (Lin et al., bioRxiv 2025) is a niche-aware GP-VAE for spatial perturbation prediction. Its
released repo has **no scoring code** — this benchmark provides it. CONCERT trains/predicts on a GPU
in its own conda env (incompatible with spbench's), so we run it **offline** and score its output
here. Pieces (all in this repo):

- `spbench/adapters/concert_export.py` — `export_to_concert_h5(data, path, counts=...)`: our
  `StandardData` → CONCERT's `.h5` (`/X` raw int counts, `/pos` 2×N, `/perturbation` `/tissue`
  `/gene` as bytes; control/no-guide → `"None"`).
- `scripts/concert/patch_build_attributes.py` — makes CONCERT's `build_attributes` data-driven
  (else any non-Perturb-Map perturbation falls through to background=0).
- `spbench/models/concert_model.py` — `ConcertModel`: loads CONCERT's counterfactual `.h5ad` and
  extracts the predicted bystander niche; scores via `compare_to_baseline(..., extra={"CONCERT": ...})`.

## Server pipeline (one Saunders slice → leaderboard)

CONCERT env recipe + cluster quirks: see `CONCERT_复现与集成方案_2026-06-18.md` (numpy 1.26.4 +
chex 0.1.86, `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib`, `-w node03` no `--gres`, github blocked).

1. **Export a slice → CONCERT `.h5`** (in spbench's env — reads the Saunders h5mu):
   ```python
   from spbench.adapters import SaundersAdapter
   from spbench.adapters.concert_export import export_to_concert_h5
   data = SaundersAdapter(DIR, max_files=1, counts_layer="<RAW>").load()   # one slice + raw counts
   export_to_concert_h5(data, "saunders_slice0.h5", counts=data.meta["counts"])
   ```
   ⚠️ confirm `<RAW>` first: data.X is `layers/raw_scaled` (z-scored, not counts). `counts_layer`
   reads `mod/rna/X` (set `"X"`) or `mod/rna/layers/<name>`. Inspect the .h5mu on the server
   (`h5ls -r file.h5mu | grep -iE "layers|/X"`) to find the integer-count layer, then set it.

2. **Patch CONCERT** (once, on the server copy):
   ```bash
   python scripts/concert/patch_build_attributes.py ~/concertRepro/CONCERT/src/run_concert_map.py
   ```

3. **Train** (GPU, node03): same sbatch as Phase A but `--data_file saunders_slice0.h5`,
   `--tissue` now = cell_type, perturbations now = KO genes. ~tens of min depending on N cells.

4. **Counterfactual per eval perturbation** P (`--stage eval`): hold out a region; flip its matched
   control cells to P → `..._<tissue>_<P>_perturbed_counts.h5ad`. **Fair-split caveat (B2):** CONCERT
   can only predict SEEN perturbations; evaluate on held-out cells/slice (spatial generalization),
   not unseen perturbations.

5. **Score in spbench** (CPU, spbench env): for each P,
   ```python
   from spbench.models.concert_model import ConcertModel
   cm = ConcertModel({P: f"out/..._{P}_perturbed_counts.h5ad"})
   concert_niche = cm.predict_niche(data, P, edges)
   res = compare_to_baseline(niches_for_P, residuals=resid, extra={"CONCERT": concert_niche})
   ```
   → CONCERT lands on the same gain / PCC-delta leaderboard as TrivialSeed/Gaussian/GCN + e_null/oracle.
   Headline question: **does CONCERT beat the no-effect baseline and our GCN on our data?**

   Preferred wiring (whole benchmark in one call, niche second board): pass CONCERT as an
   external/end-to-end model to `run_benchmark` and read the single PCC-delta currency off
   `summary_table` (the `niche_CONCERT` column):
   ```python
   from spbench.config import run_benchmark
   from spbench.plotting import summary_table
   cms = {P: ConcertModel({P: f"out/..._{P}_perturbed_counts.h5ad"}) for P in perts}
   # one ConcertModel per P, or a single model whose dict holds every P:
   res = run_benchmark(data, perturbations=perts,
                       external_models={"CONCERT": ConcertModel(all_paths)}, progress=False)
   for row in summary_table(res):
       print(row["guide"], row.get("niche_CONCERT"))   # PCC-delta, higher = better
   ```
   This is the same path the regression test `tests/test_concert_in_run_benchmark.py` exercises
   (CONCERT → `res["compare"][P]["pcc"]["CONCERT"]` → `niche_CONCERT`). The `--stage eval`
   counterfactual `.h5ad` layout (`/X` dense or CSR) is what `ConcertModel.read_h5ad_X` reads.

## Load-bearing pre-flight (confirm before any train run)

`h5ls -r <saunders>.h5mu | grep -iE "layers|/X"` to locate the **raw integer counts** layer
(`mod/rna/X` → `counts_layer="X"`, or `mod/rna/layers/<name>`). `data.X` is `layers/raw_scaled`
(z-scored) and is NOT counts — exporting it silently trains CONCERT on the wrong space. This is the
one step that, if skipped, invalidates the whole run.
