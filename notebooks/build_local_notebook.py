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
     "from spbench.viz import (plot_2x2, plot_baseline_gain, plot_gain_per_perturbation,\n"
     "                         plot_significance_contrast, plot_learned_value)")
md("## 1. Generate synthetic data (planted seed + propagation effects)")
code("data = make_synthetic(seed=0)\n"
     "print('cells', data.n_cells, '| genes', data.n_genes,\n"
     "      '| perturbations', data.perturbations())")
md("## 2. Run the benchmark (trivial seed + Gaussian baseline + simple GCN)")
code("PERTS = data.perturbations()\n"
     "res = run_benchmark(data, perturbations=PERTS, k=15, k_ref=5,\n"
     "                    gcn_kwargs={'hidden': 32, 'epochs': 20})")
md("## 3. Energy-distance + gain table (per perturbation)\n"
   "All distances are to the observed niche (lower = closer). `e_null` = no-effect baseline, "
   "`oracle` = ceiling, `m+lrn` = deployable (model seed + learned prop). "
   "**`gain = e_null − e[model+learned]` > 0 means the deployable pipeline beats doing nothing.**")
code("hdr = f\"{'pert':6} {'e_null':>7} {'oracle':>7} {'GT+base':>8} {'GT+lrn':>7} {'m+base':>7} {'m+lrn':>7} {'gain':>7}\"\n"
     "print(hdr); print('-' * len(hdr))\n"
     "for p in PERTS:\n"
     "    e = res['compare'][p]['e']; g = res['compare'][p]['gain']\n"
     "    print(f\"{p:6} {e['null']:7.3f} {e.get('oracle', float('nan')):7.3f} \"\n"
     "          f\"{e['GT+base']:8.3f} {e['GT+learned']:7.3f} {e['model+base']:7.3f} \"\n"
     "          f\"{e['model+learned']:7.3f} {g['model+learned']:7.3f}\")")
md("## 4. Result figures\n"
   "On synthetic data the seed is planted on gene i and the propagation on gene i+10, so models "
   "that 'spread the seed' predict the wrong gene — `gain` is correctly **negative** (worse than "
   "no effect) while the `oracle` ceiling is positive (there IS recoverable signal). That is the "
   "intended sanity check, not a bug. We mark a subset as significant just to demonstrate the "
   "colours; on real data pass the MC-spatial significant list.")
code("SIGNIFICANT = PERTS[:2]   # demo only; on real data use the MC-spatial significant list\n"
     "plot_baseline_gain(res)                        # headline: each method vs the baseline")
code("plot_gain_per_perturbation(res, SIGNIFICANT)   # per gene: deployable model vs baseline")
code("plot_significance_contrast(res, SIGNIFICANT)   # is the GCN-vs-Gaussian edge signal-specific?")
code("plot_learned_value(res, SIGNIFICANT)           # learned_value (e1-e2) per perturbation")
md("**Single-gene 2x2 (explainer, not a result):**")
code("plot_2x2(res['grids'][PERTS[0]], title=PERTS[0])")
md("## 5. Bottom line (how many beat the no-effect baseline)")
code("g = {p: res['compare'][p]['gain']['model+learned'] for p in PERTS}\n"
     "print('beat baseline (gain>0):', [p for p in PERTS if g[p] > 0])\n"
     "print('ranking by end-to-end E-distance:', res['ranking'])")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "run_local_synthetic.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)
