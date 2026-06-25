# CPA offline env — recipe (zenglab cluster)

The CPA runner (`run_cpa.py`) needs its **own** conda env: `cpa-tools` pins a recent scvi/jax stack
that conflicts with both the shared `.venv` and the `scgen` env (which is pinned to the 2022-era
scvi 0.20). Keep CPA isolated. CPA requires **python < 3.11** (cpa-tools / its scvi-tools pin do not
build cleanly on 3.11+).

## Build (one-off)

```bash
# 1. base env (py<3.11) + CPA (pulls a compatible scvi-tools/torch automatically)
conda create -y -p $HOME/.conda/envs/cpa python=3.10
conda activate $HOME/.conda/envs/cpa
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cpa-tools scvi-tools

# 2. libstdc++ for torch (env has none -> torch can't find CXXABI_1.3.8)
conda install -p $HOME/.conda/envs/cpa -c conda-forge "libstdcxx-ng>=12" -y

# 3. scanpy for h5ad IO + numpy<2 (scvi/jax stack is not numpy-2 clean)
pip install -i <mirror> scanpy "numpy<2"
```

After install, `python -c "import cpa, scanpy"` must succeed in the cpa env (and must FAIL in the
shared venv — `run_cpa.py`'s pure functions are import-safe there, the `cpa`/`scanpy` imports are
**lazy inside `main()`**).

## Runtime env (every invocation)

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # CXXABI / GLIBC for torch
export CUDA_VISIBLE_DEVICES=""                              # force CPU if the card's CUDA differs
```

Models are small (panel-sized G, ≤ a few thousand cells). **Must run on a compute node** (node03),
not login (login has old GLIBC). On the cluster: `srun --jobid=<your jlab job> --overlap`.

## Train + predict call sequence (the runner's `main()`)

```python
import cpa, scanpy as sc
adata = sc.read_h5ad("{P}.h5ad")          # raw counts, obs['condition'] control/stimulated, obs['cell_type']

cpa.CPA.setup_anndata(
    adata, perturbation_key="condition", control_group="control",
    dosage_key=None, categorical_covariate_keys=["cell_type"], is_count_data=True,
)
model = cpa.CPA(adata)
model.train(max_epochs=200, early_stopping=True)

# Counterfactual: shift the CONTROL cells to 'stimulated' and predict.
adata_ctrl = adata[adata.obs["condition"] == "control"].copy()
adata_ctrl.obs["condition"] = "stimulated"
model.predict(adata_ctrl)
```

### Load-bearing gotcha — the counterfactual lands in `obsm['CPA_pred']`, NOT `.X`

`model.predict(adata_ctrl)` does **not** overwrite `adata_ctrl.X`. It writes the predicted
counterfactual expression matrix into **`adata_ctrl.obsm['CPA_pred']`** (an `(n_cells, G)` array).
Reading `.X` here gives you the *original control* counts, silently producing a no-effect seed. The
runner reads `pred = adata_ctrl.obsm['CPA_pred']`, then averages per cell type into the profile bank
that `aggregate_control_predict` maps onto the scored centers.

## Two-step invocation (venv export → cpa env runner)

```bash
# 1. (shared .venv) export the per-perturbation h5ad, THEN write the centers npz.
#    export_to_cpa_h5 writes ONLY {P}.h5ad; the {P}_centers.npz the runner's --centers
#    needs is a SEPARATE np.savez (same derivation as scripts/scgen/export_dumps.py).
python -c "
import numpy as np
from spbench.adapters.cpa_export import export_to_cpa_h5
from spbench.adapters.counts_export import build_counts_X
export_to_cpa_h5(data, 'P0', build_counts_X(data), 'dumps/cpa/P0.h5ad')
centers = np.where(data.perturbation == 'P0')[0]               # StandardData indices of P0's perturbed cells
np.savez('dumps/cpa/P0_centers.npz',
         center_idx=centers.astype(np.int64),
         cell_type=np.asarray([str(c) for c in data.cell_type[centers]]))
"
#    -> dumps/cpa/P0.h5ad        (from export_to_cpa_h5)
#    -> dumps/cpa/P0_centers.npz ('center_idx','cell_type' in centers order; from the np.savez above)
#
#    scripts/scgen/export_dumps.py is the canonical centers-npz producer (same
#    np.where(data.perturbation == P)[0] + data.cell_type[centers] derivation, looped over
#    perturbations) — reuse it when exporting many perturbations at once.

# 2. (cpa conda env) train + predict + dump the aligned seed
$HOME/.conda/envs/cpa/bin/python scripts/cpa/run_cpa.py \
    --h5ad   dumps/cpa/P0.h5ad \
    --centers dumps/cpa/P0_centers.npz \
    --out     dumps/cpa/P0_seed.h5ad
#    -> dumps/cpa/P0_seed.h5ad (/X = (n_centers, G), /obs/center_idx) for SeedDumpModel('cpa', ...)
```

`{P}_seed.h5ad` is the same dump contract `scripts/_dump_io.py::write_seed_dump` writes for every
offline runner (scgen, concert, cpa); `SeedDumpModel` reads it back in the shared venv.
