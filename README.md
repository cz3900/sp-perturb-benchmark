# sp-perturb-benchmark

A pluggable benchmark for **spatial single-cell perturbation prediction** — does a model
correctly predict how a genetic perturbation (CRISPR knockout) reshapes gene expression in
intact tissue, both in the perturbed cell itself and in its spatial neighbourhood?

The whole pipeline runs locally on a built-in **synthetic data generator** (no real data
required); the same pipeline runs on real spatial CRISPR data through a dataset adapter.

---

## Background

Spatial perturbation transcriptomics (e.g. Perturb-Multi / Saunders et al., *Cell* 2025;
Dhainaut 2022; Shen 2024; Binan 2025) applies pooled CRISPR perturbations *in situ* and reads
out expression with spatial coordinates. Predicting perturbation effects in this setting is
hard for three reasons:

- **No paired before/after.** Sequencing/imaging is destructive — the same cell is never
  measured both perturbed and unperturbed. Effects must be inferred at the population level.
- **Extreme label sparsity.** In real data ~99% of cells carry no detectable guide. Those
  cells are not noise — they are the spatial context, so they cannot be filtered out.
- **No perfect control.** Perturbation locations are uncontrolled; physically pairing a
  perturbed niche with a matched control niche is unreliable.

This benchmark sidesteps the "perfect control" requirement by combining four ideas that do
not need cell-level pairing:

1. **Distribution-level scoring** — compare *groups* of cells with energy distance, not
   cell-to-cell.
2. **Observed perturbed cells as ground truth** — never a synthetic counterfactual.
3. **A control-based reference niche** — propagation starts from a control state, so a
   prediction can never trivially reproduce the observed (post-perturbation) niche.
4. **Error attribution** — separate "did the cell-intrinsic part fail" from "did the spatial
   propagation fail".

---

## Method: the 2×2 (seed × propagation)

A perturbation effect is split into two steps:

- **seed** — how the perturbed cell's own expression changes.
- **propagation** — how that change spreads to neighbouring cells (the niche).

The evaluation crosses *where the seed comes from* with *how propagation is modelled*:

|                | Baseline propagation (Gaussian kernel) | Learned propagation (GNN) |
| -------------- | -------------------------------------- | ------------------------- |
| **GT seed**    | (1) upper bound for baseline prop      | (2) learned-prop capacity, leakage check |
| **Model seed** | (3) isolates seed quality              | (4) **end-to-end deployable score** |

Each cell is scored by **energy distance** between the predicted neighbour distribution and
the observed perturbed-niche distribution:

```
E^2(X, Y) = 2 * mean||X - Y||  -  mean||X - X'||  -  mean||Y - Y'||
            (between-group)        (within X)         (within Y)
```
`X` = predicted cells, `Y` = observed cells. Identical distributions → 0; no pairing needed.

Reading the grid:

- `(3) - (1)` → **seed cost**: how much is lost by predicting the seed instead of knowing it.
- `(1) - (2)` → **learned value**: how much the learned GNN beats the Gaussian baseline.
- `(4)` → the real, deployable end-to-end score.
- **Leakage audit**: if a GT-seed cell is ≈0, propagation reproduced the observed niche →
  the model is leaking. Both GT-seed cells must be clearly non-zero.

Additional metrics: `rho_niche` (cross-niche response correlation, the go/no-go signal) and
`moran_gap` (spatial-structure realism).

---

## What's in the box

```
spbench/
  data.py            StandardData — the universal internal representation
  synthetic.py       synthetic data with planted seed + propagation effects (test fixture)
  graph.py           within-slice kNN spatial graph
  split.py           split by perturbation (seen/unseen; keeps no-guide cells in train)
  reference.py       control cells matched by cell type + expression
  propagation_gt.py  observed propagation ground truth (bystander neighbours vs reference)
  harness.py         fills the 2×2 (propagation starts from a control reference)
  judge.py           attribution, leakage gate, rho_niche gate, ranking
  metrics/           energy · rho_niche · moran   (registry — add your own)
  models/            trivial_seed · gaussian_prop · gcn_prop   (Seed/Prop/EndToEnd interfaces)
  adapters/          saunders   (DatasetAdapter — add your own)
  viz.py             2×2 heatmap + attribution figure
  config.py          run_benchmark(...) entry point + YAML config
notebooks/
  run_local_synthetic.ipynb   run locally on synthetic data (no server, no real data)
  run_benchmark.ipynb         run on real Saunders .h5mu (lab server)
```

---

## Install

```bash
git clone https://github.com/cz3900/sp-perturb-benchmark.git
cd sp-perturb-benchmark
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m pytest -q          # 31 passed — all on synthetic data
```

Dependencies are light: numpy, scipy, scikit-learn, h5py, torch, matplotlib, pyyaml. The GNN
is plain PyTorch (no torch_geometric); Moran's I is implemented from scratch (no squidpy).

---

## Run locally (no data needed)

The synthetic generator *is* the data — it plants known seed and propagation effects so the
whole pipeline is testable offline.

```bash
.venv/bin/python - <<'PY'
from spbench.synthetic import make_synthetic
from spbench.config import run_benchmark
d = make_synthetic(seed=0)
res = run_benchmark(d, perturbations=["P0", "P1", "P2"], k=15, k_ref=5,
                    gcn_kwargs={"hidden": 32, "epochs": 20})
for p in res["grids"]:
    g, a = res["grids"][p], res["attribution"][p]
    print(p, {k: round(g[k]["energy_prop"], 3) for k in "1234"},
          "| learned_value", round(a["learned_value"], 3), "leak_ok", res["leakage_pass"][p])
print("ranking:", res["ranking"])
PY
```

Or open `notebooks/run_local_synthetic.ipynb` in Jupyter for the metrics table plus figures.

> On synthetic data `learned_value` is often negative — the tiny GCN does not beat the
> Gaussian baseline. That is a legitimate benchmark finding (simple baselines are strong),
> not a bug; surfacing it is the point of the 2×2.

---

## Run on real data

`notebooks/run_benchmark.ipynb` reads real spatial CRISPR `.h5mu` via the Saunders adapter and
produces the same metrics table and figures. Point it at your data:

```python
from spbench.adapters import get_adapter
from spbench.config import run_benchmark

data = get_adapter("saunders")("/path/to/Saunders_2025_40513557", max_files=4).load()
res = run_benchmark(data, perturbations=data.perturbations()[:10], k=15, k_ref=5)
```

---

## Extending (the three plug points)

The fixed harness never changes; you add small plugin files.

- **New dataset / format** → subclass `DatasetAdapter` in `spbench/adapters/` and normalize
  any format into a `StandardData` (expression, coords, perturbation, cell type, batch).
- **New model** → subclass `SeedModel`, `PropModel`, or `EndToEndModel` in `spbench/models/`
  and `@register` it.
- **New metric** → subclass `Metric` in `spbench/metrics/` and `register(...)` it.

---

## Status & roadmap

**Implemented (MVP):** the full 2×2 with a trivial-seed floor, a Gaussian-kernel baseline, a
self-supervised GCN, energy-distance scoring, leakage audit, attribution, ranking, and figures
— all verified end-to-end on synthetic data.

**Planned:**
- Unseen-perturbation seed predictors beyond the trivial floor (nearest-gene, literature/GRN
  gene embeddings, foundation-model transfer).
- `rho_niche` no-niche ablation wired to the +0.10 go/no-go gate.
- End-to-end adapters for competing methods (CONCERT, SpatialProp).
- Significant-perturbation screening (Monte-Carlo permutation) to define the evaluation set.
- Cross-dataset runs over all collected spatial CRISPR datasets.
