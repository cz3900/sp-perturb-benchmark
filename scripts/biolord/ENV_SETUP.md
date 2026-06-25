# biolord offline env — recipe (zenglab cluster)

The biolord runner (`run_biolord.py`) needs its **own** conda env: `biolord` pins a recent
scvi-tools / jax stack that conflicts with both the shared `.venv` and the `scgen` env (pinned to the
2022-era scvi 0.20) and with the `cpa` env. Keep biolord isolated.

## Build (one-off)

```bash
# 1. base env + biolord (pulls a compatible scvi-tools / jax / torch automatically)
conda create -y -p $HOME/.conda/envs/biolord python=3.10
conda activate $HOME/.conda/envs/biolord
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple biolord scvi-tools jax

# 2. libstdc++ for torch/jax (env has none -> can't find CXXABI_1.3.8)
conda install -p $HOME/.conda/envs/biolord -c conda-forge "libstdcxx-ng>=12" -y

# 3. scanpy for h5ad IO + numpy<2 (scvi/jax stack is not numpy-2 clean)
pip install -i <mirror> scanpy "numpy<2"
```

After install, `python -c "import biolord, scanpy"` must succeed in the biolord env (and must FAIL in
the shared venv — `run_biolord.py`'s pure functions are import-safe there, the `biolord`/`scanpy`
imports are **lazy inside `main()`**).

## Runtime env (every invocation)

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # CXXABI / GLIBC for torch/jax
export CUDA_VISIBLE_DEVICES=""                              # force CPU if the card's CUDA differs
```

Models are small (panel-sized G, ≤ a few thousand cells). **Must run on a compute node** (node03),
not login (login has old GLIBC). On the cluster: `srun --jobid=<your jlab job> --overlap`.

## Train + counterfactual call sequence (the runner's `main()`)

```python
import biolord, scanpy as sc
adata = sc.read_h5ad("{P}.h5ad")          # raw counts, obs['condition'] control/stimulated, obs['cell_type']

biolord.Biolord.setup_anndata(adata, categorical_attributes_keys=["condition", "cell_type"])
model = biolord.Biolord(adata, n_latent=32)
model.train(max_epochs=200, early_stopping=True)

# Counterfactual: take the CONTROL cells and decode them with condition OVERRIDDEN to 'stimulated'.
adata_ctrl = adata[adata.obs["condition"] == "control"].copy()
pred = model.compute_prediction_adata(adata_ctrl, target_attributes={"condition": "stimulated"})
```

### Load-bearing gotcha — the counterfactual is a CONDITION TENSOR-DICT OVERRIDE, not the default forward

biolord disentangles each cell into an **ordered/unknown latent** plus the **known attribute
embeddings** (here `condition` and `cell_type`). The model's *default* forward is **reconstruction**:
it decodes a cell back to its own observed condition. Reading that output for a control cell gives you
the control state back — a silent **no-effect seed**.

The counterfactual must instead **manually override the condition entry of the attribute tensor-dict**
to the perturbed label and decode that:

1. Encode the control cells -> get their latent + attribute tensor-dict.
2. **Replace the `condition` tensor** in that dict with the embedding/index of `'stimulated'`
   (leave the latent and `cell_type` untouched — this is what isolates the condition effect).
3. Run the **decoder** on the overridden dict -> the perturbed expression `pred` `(n_ctrl, G)`.

`biolord` exposes this as `model.compute_prediction_adata(adata_ctrl,
target_attributes={"condition": "stimulated"})` (it builds the overridden tensor-dict and decodes for
you). If a given biolord version lacks that helper, do the three steps by hand against
`model.module` — the key invariant is **swap the condition tensor before decoding**; do NOT read the
plain reconstruction. The runner then averages `pred` per cell type into the profile bank that
`aggregate_control_predict` maps onto the scored centers (key `None` = global-mean fallback).

## Two-step invocation (venv export → biolord env runner)

```bash
# 1. (shared .venv) export the per-perturbation h5ad + centers npz
python -c "from spbench.adapters.biolord_export import export_to_biolord_h5; \
           from spbench.adapters.counts_export import build_counts_X; \
           export_to_biolord_h5(data, 'P0', build_counts_X(data), 'dumps/biolord/P0.h5ad')"
#    -> dumps/biolord/P0.h5ad  +  dumps/biolord/P0_centers.npz ('center_idx','cell_type')

# 2. (biolord conda env) train + counterfactual override + dump the aligned seed
$HOME/.conda/envs/biolord/bin/python scripts/biolord/run_biolord.py \
    --h5ad    dumps/biolord/P0.h5ad \
    --centers dumps/biolord/P0_centers.npz \
    --out     dumps/biolord/P0_seed.h5ad
#    -> dumps/biolord/P0_seed.h5ad (/X = (n_centers, G), /obs/center_idx) for SeedDumpModel('biolord', ...)
```

`{P}_seed.h5ad` is the same dump contract `scripts/_dump_io.py::write_seed_dump` writes for every
offline runner (scgen, concert, cpa, biolord); `SeedDumpModel` reads it back in the shared venv.
