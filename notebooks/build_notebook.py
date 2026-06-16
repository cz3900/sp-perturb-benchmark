import json, os

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": t})

md("# Spatial perturbation benchmark — run\n"
   "Runs the 2x2 (seed x propagation) benchmark, prints the metric tables, and draws figures.\n"
   "Run on the lab server (env `spatial-tumor-ai`) so it can read the real .h5mu.")
code("import matplotlib\n%matplotlib inline\n"
     "from spbench.adapters import get_adapter\n"
     "from spbench.config import run_benchmark\n"
     "from spbench.viz import plot_2x2, plot_attribution\n"
     "import numpy as np, pandas as pd")
md("## 1. Load data via adapter (swap adapter/path here to change dataset)")
code("DIR='/home/yiru/database/spatial_perturbed_processed/CRISPR_based/Saunders_2025_40513557'\n"
     "data = get_adapter('saunders')(DIR, max_files=4).load()\n"
     "print(data.n_cells, data.n_genes, 'perturbations:', len(data.perturbations()))")
md("## 2. Run the benchmark (trivial seed + Gaussian + GCN)")
code("PERTS = data.perturbations()[:10]   # later: replace with the 14 MC-spatial significant\n"
     "res = run_benchmark(data, perturbations=PERTS, k=15, k_ref=5,\n"
     "                    gcn_kwargs={'hidden':64,'epochs':30})")
md("## 3. Metric table (per-perturbation 2x2 + attribution + leakage)")
code("rows=[]\n"
     "for p in PERTS:\n"
     "    g=res['grids'][p]; a=res['attribution'][p]\n"
     "    rows.append(dict(perturbation=p, e1=g['1']['energy_prop'], e2=g['2']['energy_prop'],\n"
     "                     e3=g['3']['energy_prop'], e4=g['4']['energy_prop'],\n"
     "                     seed_cost=a['seed_cost'], learned_value=a['learned_value'],\n"
     "                     end_to_end=a['end_to_end'], leak_ok=res['leakage_pass'][p]))\n"
     "df=pd.DataFrame(rows).sort_values('end_to_end'); df")
md("## 4. Figures")
code("fig=plot_2x2(res['grids'][PERTS[0]], title=PERTS[0]); fig.savefig('fig_2x2.png', dpi=150)")
code("fig=plot_attribution(res['attribution']); fig.savefig('fig_attribution.png', dpi=150)")
md("## 5. Ranking + go/no-go note\n"
   "Lowest end-to-end E-distance = best. Add the rho_niche (with vs without niche) here once the\n"
   "no-niche variant is wired, and check the +0.10 gate.")
code("print('ranking (best first):', res['ranking'])")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "run_benchmark.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)
