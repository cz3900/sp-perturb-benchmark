import json, os

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": t})

md("# Spatial perturbation benchmark — run\n"
   "Runs the 2x2 (seed x propagation) benchmark, prints the metric table, and draws the result\n"
   "figures. Run on the lab server (env `spatial-tumor-ai`) so it can read the real .h5mu.")
code("import matplotlib\n%matplotlib inline\n"
     "from spbench.adapters import get_adapter\n"
     "from spbench.config import run_benchmark\n"
     "from spbench.viz import (plot_2x2, plot_attribution, plot_learned_value,\n"
     "                         plot_significance_contrast, plot_slope, plot_seed_vs_learned)\n"
     "import numpy as np, pandas as pd")
md("## 1. Load data via adapter (swap adapter/path here to change dataset)")
code("DIR='/home/yiru/database/spatial_perturbed_processed/CRISPR_based/Saunders_2025_40513557'\n"
     "data = get_adapter('saunders')(DIR, max_files=4).load()\n"
     "print(data.n_cells, data.n_genes, 'perturbations:', len(data.perturbations()))")
md("## 2. Define the evaluation set\n"
   "`SIGNIFICANT` = the perturbations MC-spatial flagged as having real spatial effects "
   "(p<0.05). `NON_SIGNIFICANT` = a random sample of the *same size* drawn from every other "
   "perturbation (a proxy negative-control group — the vast majority have no real spatial "
   "signal). This balanced contrast checks the learned model does not just help everywhere "
   "regardless of signal.")
code("SIGNIFICANT = ['Pck1','Rrbp1','Hspd1','Psmc1','Sepp1','Bcl2l1','Vcp',\n"
     "               'Ass1','Pten','Rrn3','Letm1','Hspa5','Sec61b','Rngtt']\n"
     "sig = [p for p in SIGNIFICANT if p in set(data.perturbations())]\n"
     "others = [p for p in data.perturbations() if p not in set(SIGNIFICANT)]\n"
     "rng = np.random.default_rng(0)   # fixed seed for reproducibility\n"
     "NON_SIGNIFICANT = list(rng.choice(others, size=min(len(sig), len(others)), replace=False))\n"
     "EVAL = sig + NON_SIGNIFICANT\n"
     "print('evaluating', len(EVAL), '=', len(sig), 'significant +',\n"
     "      len(NON_SIGNIFICANT), 'random non-significant')")
md("## 3. Run the benchmark (trivial seed + Gaussian baseline + GCN)")
code("res = run_benchmark(data, perturbations=EVAL, k=15, k_ref=5,\n"
     "                    gcn_kwargs={'hidden':64,'epochs':30})")
md("## 4. Metric table (per-perturbation 2x2 + attribution + leakage)")
code("rows=[]\n"
     "for p in EVAL:\n"
     "    g=res['grids'][p]; a=res['attribution'][p]\n"
     "    rows.append(dict(perturbation=p, sig=p in set(SIGNIFICANT),\n"
     "                     e1=g['1']['energy_prop'], e2=g['2']['energy_prop'],\n"
     "                     e3=g['3']['energy_prop'], e4=g['4']['energy_prop'],\n"
     "                     seed_cost=a['seed_cost'], learned_value=a['learned_value'],\n"
     "                     end_to_end=a['end_to_end'], leak_ok=res['leakage_pass'][p]))\n"
     "df=pd.DataFrame(rows).sort_values('end_to_end'); df")
md("## 5. Result figures\n"
   "Absolute E-distance is background-dominated (~6); the signal is in the **differences**. "
   "These figures plot the differences across perturbations and contrast significant vs "
   "non-significant — that is what reveals whether learned propagation captures real signal.")
md("**B — the headline:** learned_value distribution, significant vs non-significant "
   "(box + points + sign-test). If the significant group is not clearly higher, the GCN's gain "
   "is generic (a better niche smoother), not signal-specific.")
code("plot_significance_contrast(res, SIGNIFICANT).savefig('fig_B_contrast.png', dpi=140)")
md("**A — per perturbation:** learned_value sorted, coloured by significance (>0 = GCN beats Gaussian).")
code("plot_learned_value(res, SIGNIFICANT).savefig('fig_A_learned_value.png', dpi=140)")
md("**C — consistency + outliers:** baseline (e1) -> learned (e2) slope per perturbation "
   "(downward = GCN better). Outliers sit far from the cluster.")
code("plot_slope(res, SIGNIFICANT).savefig('fig_C_slope.png', dpi=140)")
md("**D — attribution scatter:** seed_cost vs learned_value (where does the error live?).")
code("plot_seed_vs_learned(res, SIGNIFICANT).savefig('fig_D_seed_vs_learned.png', dpi=140)")
md("**Single-gene 2x2 (explainer, not a result):** what one perturbation's grid looks like.")
code("plot_2x2(res['grids'][EVAL[0]], title=EVAL[0])")
md("## 6. Ranking + go/no-go note\n"
   "Lowest end-to-end E-distance = best. The real go/no-go is the rho_niche (with vs without "
   "niche) +0.10 gate — wire that in once the no-niche variant exists. Note: learned_value > 0 "
   "alone does **not** prove signal-specificity; compare the two groups in figure B, and add a "
   "label-permutation control.")
code("print('ranking (best first):', res['ranking'])")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "run_benchmark.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)
