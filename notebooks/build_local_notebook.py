import json, os

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": t})

md("# Spatial perturbation benchmark — LOCAL synthetic run\n"
   "Runs the full 2x2 (seed x propagation) benchmark on **synthetic data** — no server, no real "
   ".h5mu, no extra deps (pandas not required). Use this to develop/inspect the pipeline locally.\n"
   "The server version (`run_benchmark.ipynb`) is identical except its data source is the "
   "Saunders `.h5mu` adapter instead of the synthetic generator.")
code("%matplotlib inline\n"
     "from spbench.synthetic import make_synthetic\n"
     "from spbench.config import run_benchmark\n"
     "from spbench.viz import plot_2x2, plot_attribution")
md("## 1. Generate synthetic data (planted seed + propagation effects)")
code("data = make_synthetic(seed=0)\n"
     "print('cells', data.n_cells, '| genes', data.n_genes,\n"
     "      '| perturbations', data.perturbations())")
md("## 2. Run the benchmark (trivial seed + Gaussian baseline + simple GCN)")
code("PERTS = data.perturbations()\n"
     "res = run_benchmark(data, perturbations=PERTS, k=15, k_ref=5,\n"
     "                    gcn_kwargs={'hidden': 32, 'epochs': 20})")
md("## 3. Metrics table (per-perturbation 2x2 + attribution + leakage)\n"
   "Cells: (1) GT seed + baseline, (2) GT seed + learned, (3) model seed + baseline, "
   "(4) model seed + learned (end-to-end). Lower E-distance = better.")
code("hdr = f\"{'pert':6} {'e1':>6} {'e2':>6} {'e3':>6} {'e4':>6} {'seed_cost':>10} {'learned_val':>12} {'leak_ok':>8}\"\n"
     "print(hdr); print('-' * len(hdr))\n"
     "for p in PERTS:\n"
     "    g = res['grids'][p]; a = res['attribution'][p]\n"
     "    print(f\"{p:6} {g['1']['energy_prop']:6.3f} {g['2']['energy_prop']:6.3f} \"\n"
     "          f\"{g['3']['energy_prop']:6.3f} {g['4']['energy_prop']:6.3f} \"\n"
     "          f\"{a['seed_cost']:10.3f} {a['learned_value']:12.3f} {str(res['leakage_pass'][p]):>8}\")")
md("## 4. Figures")
code("plot_2x2(res['grids'][PERTS[0]], title=PERTS[0])")
code("plot_attribution(res['attribution'])")
md("## 5. Ranking (lowest end-to-end E-distance = best)\n"
   "Note: on synthetic data `learned_value` is often negative (the tiny GCN does not beat the "
   "Gaussian baseline) — that is a legitimate benchmark finding, not a bug.")
code("print('ranking (best first):', res['ranking'])")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "run_local_synthetic.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)
