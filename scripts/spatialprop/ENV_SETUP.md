# SpatialProp offline env — recipe (zenglab cluster)

The SpatialProp runner (`run_spatialprop.py`) needs its **own** conda env: `spatialprop` pins a
torch 2.6 / PyG 2.6 stack that conflicts with both the shared `.venv` and the older offline envs
(`scgen` is pinned to scvi 0.20, `cpa` needs python < 3.11 + a different scvi/jax line). Keep
SpatialProp isolated.

**No pre-trained weights ship — training is from scratch.** Each run builds and trains a fresh
SpatialProp model on the exported slice (panel-sized G, ≤ a few thousand cells), so there is nothing
to download; the cost is the per-perturbation training pass.

## Build (one-off)

```bash
# 1. base env + torch 2.6 (CPU or matching-CUDA wheel)
conda create -y -p $HOME/.conda/envs/spatialprop python=3.11
conda activate $HOME/.conda/envs/spatialprop
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "torch==2.6.*"

# 2. PyG 2.6 + its compiled companions, built against the torch 2.6 ABI
pip install -i <mirror> "torch_geometric==2.6.*"
pip install pyg_lib torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.6.0+cpu.html   # match your torch/CUDA build tag

# 3. SpatialProp itself + scanpy for h5ad IO
pip install -i <mirror> spatialprop scanpy

# 4. libstdc++ for torch (env has none -> torch can't find CXXABI_1.3.8)
conda install -p $HOME/.conda/envs/spatialprop -c conda-forge "libstdcxx-ng>=12" -y
```

After install, `python -c "import spatialprop, scanpy"` must succeed in the spatialprop env (and must
FAIL in the shared venv — `run_spatialprop.py`'s `build_multiplier_matrix` is import-safe there, the
`spatialprop`/`scanpy` imports are **lazy inside `main()`**).

## Runtime env (every invocation)

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # CXXABI / GLIBC for torch
export CUDA_VISIBLE_DEVICES=""                              # force CPU if the card's CUDA differs
```

Models are small. **Must run on a compute node** (node03), not login (login has old GLIBC). On the
cluster: `srun --jobid=<your jlab job> --overlap`.

## Train + predict call sequence (the runner's `main()`)

```python
import scanpy as sc
from spatialprop import (train_perturbation_model, create_perturbation_input_matrix,
                         predict_perturbation_effects)

adata = sc.read_h5ad("{P}.h5ad")                  # exported slice, var['gene_names'], obs['celltype']

# 1. train a SpatialProp perturbation model from scratch on this slice
model = train_perturbation_model(adata)

# 2. build the per-celltype gene-multiplier input for the guide (in-panel only, see below)
mult  = build_multiplier_matrix("{P}", genes, celltypes, value=0.0)   # rows=celltypes, cols=genes
inp   = create_perturbation_input_matrix(adata, mult)

# 3. predict the counterfactual and stash it in the output layer
pred  = predict_perturbation_effects(model, inp)
adata.layers["predicted_tempered"] = pred
adata.write_h5ad("{P}.out.h5ad")
```

### The multiplier input — per-celltype gene multipliers

`build_multiplier_matrix(guide, genes, celltypes, value=0.0)` returns an `(n_celltypes, n_genes)`
matrix: every cell is 1.0 (gene left untouched) except the guide gene's column, which is set to
`value` (0.0 for a knockout). `create_perturbation_input_matrix` expands that per-celltype multiplier
onto the cells. The value is a **multiplier**, not an absolute level: 0.0 = full knockout, 0.5 = 50%
knockdown, etc.

### Load-bearing gotcha — the counterfactual lands in `layers['predicted_tempered']`

`predict_perturbation_effects` returns the tempered counterfactual expression matrix, which the runner
writes into **`adata.layers['predicted_tempered']`** (an `(n_cells, G)` array). This is exactly the
slot `SpatialPropModel` (Task 4.2) reads back in the shared venv. Do NOT overwrite `.X` — the loader
keys on `layers['predicted_tempered']`, and a prediction left in `.X` would be silently ignored.

### Off-panel guide limitation (same limit as scGEN)

A guide gene that is **not in the measured panel** (`genes`) cannot be expressed as a multiplier
input — there is no column to scale. `build_multiplier_matrix` returns `None` for such guides, and
`main()` exits with a clear message rather than producing a no-effect prediction. This mirrors the
scGEN off-panel limitation: the model can only perturb genes it actually measures.

## Two-step invocation (venv export → spatialprop env runner)

```bash
# 1. (shared .venv) export the per-perturbation slice h5ad (var['gene_names'], obs['celltype'])
python -c "from spbench.adapters.spatialprop_export import export_to_spatialprop_h5; \
           export_to_spatialprop_h5(data, 'P0', out_dir='dumps/spatialprop')"
#    -> dumps/spatialprop/P0.h5ad

# 2. (spatialprop conda env) train from scratch + predict + write the predicted_tempered layer
$HOME/.conda/envs/spatialprop/bin/python scripts/spatialprop/run_spatialprop.py \
    --h5ad dumps/spatialprop/P0.h5ad \
    --pert P0 \
    --out  dumps/spatialprop/P0.out.h5ad
#    -> dumps/spatialprop/P0.out.h5ad (layers['predicted_tempered']) for SpatialPropModel
```

Unlike the seed runners (scgen, concert, cpa), this runner does **not** use the
`scripts/_dump_io.py::write_seed_dump` contract — it writes the counterfactual into
`layers['predicted_tempered']` via anndata, which is what `SpatialPropModel` reads in the shared venv.
