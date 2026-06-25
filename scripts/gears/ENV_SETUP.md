# GEARS offline env — recipe (zenglab cluster)

The GEARS runner (`run_gears.py`) needs its **own** conda env: `gears` pulls **PyTorch Geometric
(PyG)** plus a recent torch, which conflicts with both the shared `.venv` and the `scgen`/`cpa` envs
(2022-era scvi pins). Keep GEARS isolated. GEARS (Roohani et al. 2024, *Nat Biotechnol*) is a
GNN that propagates the perturbation signal over a **Gene Ontology (GO) similarity graph**, so the
env also needs that graph + a `gene2go` mapping pre-fetched (see the load-bearing step below).

## Build (one-off)

```bash
# 1. base env (py 3.10) + GEARS (PyPI package is `cell-gears`) + torch + PyG
conda create -y -p $HOME/.conda/envs/gears python=3.10
conda activate $HOME/.conda/envs/gears
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cell-gears torch torch-geometric

# 2. libstdc++ for torch (env has none -> torch can't find CXXABI_1.3.8)
conda install -p $HOME/.conda/envs/gears -c conda-forge "libstdcxx-ng>=12" -y

# 3. scanpy for h5ad IO + numpy<2 (PyG / the GEARS stack is not numpy-2 clean)
pip install -i <mirror> scanpy "numpy<2"
```

After install, `python -c "import gears, scanpy"` must succeed in the gears env (and is irrelevant in
the shared venv — `run_gears.py`'s `broadcast_seed` is import-safe there, the `from gears import ...`
is **lazy inside `main()`**).

## Load-bearing step — pre-download the GO graph / gene2go BEFORE the offline run

GEARS fetches its GO-derived gene-similarity graph and a `gene2go` mapping **at `PertData` /
`GEARS` init time** over the network. The cluster compute nodes have **no outbound internet**, so the
first init will hang/fail there. Pre-fetch on a node that *can* reach the network (or the login node),
into a directory the offline run will point at:

```bash
# on a network-capable node, ONCE — populates ./data/go_essential_<...>.csv, gene2go_all.pkl, etc.
python -c "
from gears import PertData
pd = PertData('./gears_data')          # downloads GO graph + gene2go into ./gears_data on first call
"
```

Keep `./gears_data` on shared storage and reuse it for every perturbation — the graph is dataset-
independent. If GEARS can't find a cached copy it silently tries to download and dies on the node.

## PertData construction

GEARS wraps the AnnData in a `PertData` object keyed on `obs['condition']` (`'GENE+ctrl'` / `'ctrl'`,
which `gears_export` writes) and `var['gene_name']` (the singular column `gears_export` writes):

```python
from gears import PertData, GEARS
import scanpy as sc
adata = sc.read_h5ad("{P}.h5ad")        # raw counts, obs['condition']='{P}+ctrl'/'ctrl', var['gene_name']
pert_data = PertData("./gears_data")    # reuses the pre-fetched GO graph (above)
pert_data.new_data_process(dataset_name="{P}", adata=adata)
pert_data.prepare_split(split="no_test", seed=1)
pert_data.get_dataloader(batch_size=32, test_batch_size=64)

model = GEARS(pert_data, device="cpu")
model.model_initialize(hidden_size=64)
model.train(epochs=20)

# GEARS predicts ONE mean expression vector per perturbation (not per cell):
pred_mean = model.predict([[ "{P}" ]])[ "{P}" ]    # (G,)
```

### Off-panel guide limit (same as scGEN)

A guide gene that is **off the panel** (not in `var['gene_name']`) cannot be a GEARS node — GEARS can
only perturb genes present in the GO graph *and* the expression matrix. Such guides are skipped, the
same off-panel limitation scGEN has; the export already restricts to the panel genes.

## Two-step invocation (venv export → gears env runner)

```bash
# 1. (shared .venv) export the per-perturbation h5ad, THEN write the centers npz.
#    export_to_gears_h5 writes ONLY {P}.h5ad; the {P}_centers.npz the runner's --centers
#    needs is a SEPARATE np.savez (same derivation as scripts/scgen/export_dumps.py).
python -c "
import numpy as np
from spbench.adapters.gears_export import export_to_gears_h5
from spbench.adapters.counts_export import build_counts_X
export_to_gears_h5(data, 'P0', build_counts_X(data), 'dumps/gears/P0.h5ad')
centers = np.where(data.perturbation == 'P0')[0]               # StandardData indices of P0's perturbed cells
np.savez('dumps/gears/P0_centers.npz',
         center_idx=centers.astype(np.int64),
         cell_type=np.asarray([str(c) for c in data.cell_type[centers]]))
"
#    -> dumps/gears/P0.h5ad        (from export_to_gears_h5)
#    -> dumps/gears/P0_centers.npz ('center_idx','cell_type' in centers order; from the np.savez above)
#
#    scripts/scgen/export_dumps.py is the canonical centers-npz producer (same
#    np.where(data.perturbation == P)[0] + data.cell_type[centers] derivation, looped over
#    perturbations) — reuse it when exporting many perturbations at once.

# 2. (gears conda env) train + predict + broadcast + dump the aligned seed
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # CXXABI / GLIBC for torch
export CUDA_VISIBLE_DEVICES=""                              # force CPU if the card's CUDA differs
$HOME/.conda/envs/gears/bin/python scripts/gears/run_gears.py \
    --h5ad    dumps/gears/P0.h5ad \
    --centers dumps/gears/P0_centers.npz \
    --pert    P0 \
    --out     dumps/gears/P0_seed.h5ad
#    -> dumps/gears/P0_seed.h5ad (/X = (n_centers, G), /obs/center_idx) for SeedDumpModel('gears', ...)
```

GEARS emits a **single per-perturbation mean** (not per cell), so the runner's `broadcast_seed` tiles
that one vector to all `n_centers` of P — every center gets the same seed. `{P}_seed.h5ad` is the same
dump contract `scripts/_dump_io.py::write_seed_dump` writes for every offline runner (scgen, concert,
cpa, gears); `SeedDumpModel` reads it back in the shared venv.

Models are small (panel-sized G, ≤ a few thousand cells). **Must run on a compute node** (node03),
not login (login has old GLIBC). On the cluster: `srun --jobid=<your jlab job> --overlap`.
