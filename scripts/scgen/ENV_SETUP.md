# scGEN offline env — exact recipe (zenglab cluster, verified 2026-06-24)

The scGEN runner (`run_scgen.py`) needs its **own** conda env (scvi/jax pins conflict with the
shared `concert`/`.venv`). scgen 2.1.0 (latest on PyPI) was written for the **2022-era scvi stack**,
so the whole dependency web has to be pinned back. This is the verified, working set — reproduce it
exactly rather than re-deriving the cascade.

## Build (one-off)

```bash
# 1. base env + scgen (pulls a too-new scvi/jax/anndata — we pin them back below)
conda create -y -p $HOME/.conda/envs/scgen python=3.10
conda activate $HOME/.conda/envs/scgen
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scgen          # gets scgen 2.1.0

# 2. libstdc++ for torch (env has none -> torch can't find CXXABI_1.3.8)
conda install -p $HOME/.conda/envs/scgen -c conda-forge "libstdcxx-ng>=12" -y

# 3. pin scvi + the whole 2022-era web it needs (order matters less than the final versions)
pip install -i <mirror> "scvi-tools==0.20.3" "numpy<2"
pip install -i <mirror> requests
pip install -i <mirror> "anndata==0.9.2" "scanpy==1.9.3" "pandas<2"
pip install -i <mirror> "mudata==0.2.3"
pip install -i <mirror> "jax==0.4.23" "jaxlib==0.4.23" "ml_dtypes==0.2.0" "flax==0.7.4" "orbax-checkpoint==0.4.4"
pip install -i <mirror> "scipy==1.10.1" "numpyro==0.13.2"
```

### Final working versions (`pip freeze`, verified end-to-end)

```
scgen==2.1.0           scvi-tools==0.20.3     torch==2.12.1          pytorch-lightning==1.9.5
anndata==0.9.2         scanpy==1.9.3          mudata==0.2.3          pandas==1.5.3  numpy==1.26.4
jax==0.4.23            jaxlib==0.4.23         flax==0.7.4            orbax-checkpoint==0.4.4
ml-dtypes==0.2.0       chex==0.1.86           scipy==1.10.1          numpyro==0.13.2
```

## 3 source patches to scgen 2.1.0 (it predates scvi 0.20's renames)

`$ENV/lib/python3.10/site-packages/scgen/_scgenvae.py`:

| line | from | to |
|---|---|---|
| 4 | `from scvi._compat import Literal` | `from typing import Literal` |
| 5 | `...import BaseModuleClass, LossRecorder, auto_move_data` | `...import BaseModuleClass, LossOutput, auto_move_data` |
| 131 | `return LossOutput(loss, rl, kld)` | `return LossOutput(loss=loss, reconstruction_loss=rl, kl_local=kld)` |

(`LossRecorder`→`LossOutput` rename; `LossOutput` is a chex mappable-dataclass → **keyword** args only.)
Then delete `scgen/__pycache__/_scgenvae.cpython-310.pyc`.

## Runtime env (every invocation)

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # CXXABI / GLIBC for torch
export CUDA_VISIBLE_DEVICES=""                              # torch is cu13, node03 driver is CUDA-12 -> force CPU
```

Models are tiny (209 genes, ≤500 cells) so CPU is fine (~9 s/guide). **Must run on a compute node**
(node03), not login (login has old GLIBC). On the cluster: `srun --jobid=<your jlab job> --overlap`.
