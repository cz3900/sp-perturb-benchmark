import json, os

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": t})

md("# Spatial perturbation benchmark — run\n"
   "\n"
   "Evaluates how well a model predicts the **spatial effect of a CRISPR perturbation**: when a "
   "gene is knocked out, how does the perturbed cell itself change (**seed**) and how does that "
   "change spread to its spatial neighbours (**propagation**)?\n"
   "\n"
   "The same cell is never measured both perturbed and unperturbed (no paired before/after), so "
   "everything is scored at the **distribution level** with the **energy distance (E-distance)**, "
   "against a **control reference niche**. Run on the lab server (env `spatial-tumor-ai`).")

md("## 0. What is measured (read this first)\n"
   "\n"
   "**Two halves of a perturbation effect**\n"
   "- **seed** = how the *perturbed cell itself* changes.\n"
   "- **propagation** = how that change spreads to its *spatial neighbours* (the niche).\n"
   "\n"
   "**The 2×2.** Every perturbation is scored in a grid of {seed source} × {propagation model}:\n"
   "\n"
   "|                | baseline prop (Gaussian kernel) | learned prop (GCN) |\n"
   "|----------------|---------------------------------|--------------------|\n"
   "| **GT seed** (the true perturbed cell, an oracle) | `e1` | `e2` |\n"
   "| **model seed** (predicted from control cells)    | `e3` | `e4` |\n"
   "\n"
   "Each `e1..e4` is an **E-distance** between the *predicted* neighbour distribution and the "
   "*observed perturbed* neighbour distribution. Lower = better. `e4` (model seed + learned prop) "
   "is the real, deployable score.\n"
   "\n"
   "**Energy distance.** For two groups of cells X (predicted) and Y (observed):\n"
   "`E = 2·mean‖X−Y‖ − mean‖X−X'‖ − mean‖Y−Y'‖`. It is 0 when the two clouds overlap and needs "
   "no cell-to-cell pairing — it compares whole distributions.\n"
   "\n"
   "**Distributional readout (why predictions get per-cell noise).** A model emits one vector "
   "per cell — its conditional *mean*. Real niches are full-variance clouds (spread ~6), so "
   "scoring a near-degenerate mean-field cloud with the energy distance (a *distributional* "
   "metric) inflates it structurally, no matter how good the mean shift is. So before scoring, "
   "each predicted cell gets a sampled **control residual** added (the deterministic analogue of "
   "a generative model drawing per-cell samples): it restores realistic per-cell variance without "
   "moving the mean, so E-distance measures the predicted *shift* fairly. Residuals come only from "
   "control cells — never the observed niche — so they cannot leak. (`distributional=True`.)\n"
   "\n"
   "**Derived numbers (computed from `e1..e4`):**\n"
   "- `seed_cost = e3 − e1` — penalty for *predicting* the seed vs knowing it (same propagation). "
   ">0 means seed prediction hurts.\n"
   "- `learned_value = e1 − e2` — how much the learned GCN beats the Gaussian baseline (same true "
   "seed). >0 means learned propagation helps.\n"
   "- `end_to_end = e4` — the deployable score.\n"
   "- `leak_ok` — leakage audit. Propagation starts from a *control reference*, so if a GT-seed "
   "cell (`e1`/`e2`) were ≈0 it would mean the model copied the observed niche (a leak). Both must "
   "be clearly > 0.\n"
   "\n"
   "**Skill (0..100%).** Absolute E-distance (~6) is dominated by background cell variability, so "
   "each perturbation is recalibrated:\n"
   "- `floor` = E-distance between two halves of the *observed* perturbed niche (pure sampling "
   "noise — the best a perfect prediction could reach).\n"
   "- `S` = E-distance(observed perturbed niche, control niche) = the total real effect to predict "
   "(the ceiling).\n"
   "- `skill = (S − model_error) / (S − floor)` — **0** = no better than predicting 'no effect', "
   "**100%** = perfect. A perturbation is kept only if `S` is **both** statistically above `floor` "
   "(z>2) **and** a meaningful effect size (gap ≥ `min_rel`·floor, default 5%); otherwise the skill "
   "ratio's denominator is tiny and ill-conditioned, so the perturbation is dropped as **no "
   "signal**.")

md("## 1. Imports")
code("import matplotlib\n%matplotlib inline\n"
     "from spbench.adapters import get_adapter\n"
     "from spbench.config import run_benchmark\n"
     "from spbench.viz import (plot_2x2, plot_attribution, plot_learned_value,\n"
     "                         plot_significance_contrast, plot_slope, plot_seed_vs_learned,\n"
     "                         plot_skill_leaderboard)\n"
     "import numpy as np, pandas as pd")

md("## 2. Load the data\n"
   "The adapter reads the raw `.h5mu` slices and normalises them into one `StandardData` "
   "(expression matrix, spatial coords, per-cell perturbation label, cell type, slice id). "
   "`max_files` limits how many slices are pooled — raise it for more cells. Swap the adapter / "
   "path here to run a different dataset.")
code("DIR='/home/yiru/database/spatial_perturbed_processed/CRISPR_based/Saunders_2025_40513557'\n"
     "data = get_adapter('saunders')(DIR, max_files=4).load()\n"
     "print(data.n_cells, 'cells,', data.n_genes, 'genes,',\n"
     "      len(data.perturbations()), 'perturbations')")

md("## 3. Define the evaluation set\n"
   "`SIGNIFICANT` = the perturbations MC-spatial flagged as having a real spatial effect "
   "(permutation p<0.05). `NON_SIGNIFICANT` = a random sample of the **same size** from every "
   "other perturbation — a proxy negative-control group (most perturbations have no spatial "
   "signal). Evaluating both, balanced, lets us check whether the model helps *specifically* "
   "where there is signal, or just everywhere (which would mean it is only a generic smoother).")
code("SIGNIFICANT = ['Pck1','Rrbp1','Hspd1','Psmc1','Sepp1','Bcl2l1','Vcp',\n"
     "               'Ass1','Pten','Rrn3','Letm1','Hspa5','Sec61b','Rngtt']\n"
     "sig = [p for p in SIGNIFICANT if p in set(data.perturbations())]\n"
     "others = [p for p in data.perturbations() if p not in set(SIGNIFICANT)]\n"
     "rng = np.random.default_rng(0)   # fixed seed -> reproducible non-significant sample\n"
     "NON_SIGNIFICANT = list(rng.choice(others, size=min(len(sig), len(others)), replace=False))\n"
     "EVAL = sig + NON_SIGNIFICANT\n"
     "print('evaluating', len(EVAL), '=', len(sig), 'significant +',\n"
     "      len(NON_SIGNIFICANT), 'random non-significant')")

md("## 4. Run the benchmark\n"
   "For each perturbation this builds the spatial graph, defines the control reference niche and "
   "the observed propagation ground truth, then fills the 2×2 by composing three models — a "
   "**trivial seed** (control + global mean shift), a **Gaussian-kernel** baseline propagation, "
   "and a self-supervised **GCN** learned propagation — and scores every cell with the "
   "E-distance. `compute_skill=True` also calibrates each perturbation (floor / S) and returns "
   "the 0..1 skill. (`k` = neighbours per cell; `k_ref` = matched control cells per perturbed "
   "cell.)")
code("res = run_benchmark(data, perturbations=EVAL, k=15, k_ref=5,\n"
     "                    gcn_kwargs={'hidden':64,'epochs':30})")

md("## 5. Metric table\n"
   "One row per perturbation. Columns: `e1..e4` are the four 2×2 E-distances (see section 0); "
   "`seed_cost = e3−e1`, `learned_value = e1−e2`, `end_to_end = e4`; `leak_ok` is the leakage "
   "audit. Sorted by `end_to_end` (best first). Remember the absolute E-distances (~6) are "
   "background-dominated — read the **differences** (`seed_cost`, `learned_value`) and the skill "
   "leaderboard below, not the raw values.")
code("rows=[]\n"
     "for p in EVAL:\n"
     "    g=res['grids'][p]; a=res['attribution'][p]; s=res['skill'].get(p, {})\n"
     "    rows.append(dict(perturbation=p, sig=p in set(SIGNIFICANT),\n"
     "                     e1=g['1']['energy_prop'], e2=g['2']['energy_prop'],\n"
     "                     e3=g['3']['energy_prop'], e4=g['4']['energy_prop'],\n"
     "                     seed_cost=a['seed_cost'], learned_value=a['learned_value'],\n"
     "                     end_to_end=a['end_to_end'], leak_ok=res['leakage_pass'][p],\n"
     "                     has_signal=s.get('has_signal'), skill_learned=s.get('learned')))\n"
     "df=pd.DataFrame(rows).sort_values('end_to_end'); df")

md("## 6. Result figures\n"
   "Absolute E-distance is background-dominated (~6); the signal lives in the **differences** and "
   "in the **calibrated skill**. The figures below aggregate across perturbations and contrast "
   "significant vs non-significant — that is what reveals whether learned propagation captures "
   "real signal rather than just smoothing better everywhere.")

md("### Headline — skill leaderboard\n"
   "Absolute E-distance turned into a **0..100% skill** = fraction of the recoverable niche "
   "signal each model captures (via `calibrate_edistance`: skill `= (S − error)/(S − floor)`). "
   "Only perturbations whose perturbed niche sits clearly above the noise `floor` are shown; the "
   "rest are dropped (no signal to predict). Bars are clipped to ±100%; **<0 = worse than "
   "predicting 'no effect'**. This is the one-glance 'how good is it' figure.")
code("plot_skill_leaderboard(res).savefig('fig_skill_leaderboard.png', dpi=140); None")

md("### B — significance contrast (the proof)\n"
   "`learned_value` (= e1−e2) for the significant group vs the non-significant group, with a "
   "sign-test. **If the significant group is not clearly higher, the GCN's advantage is generic** "
   "(a better niche smoother), not specific to real spatial signal.")
code("plot_significance_contrast(res, SIGNIFICANT).savefig('fig_B_contrast.png', dpi=140); None")

md("### A — learned_value per perturbation\n"
   "Every perturbation's `learned_value` (= e1−e2), sorted, coloured by significance. "
   "**>0 = the GCN beats the Gaussian baseline.** Outliers and the significant/non-significant "
   "split are visible at a glance.")
code("plot_learned_value(res, SIGNIFICANT).savefig('fig_A_learned_value.png', dpi=140); None")

md("### C — baseline → learned slope (consistency + outliers)\n"
   "A line per perturbation from `e1` (baseline prop) to `e2` (learned prop). **Downward = the "
   "GCN is better.** Direction-consistency and any outlier (a line far from the cluster) jump "
   "out.")
code("plot_slope(res, SIGNIFICANT).savefig('fig_C_slope.png', dpi=140); None")

md("### D — attribution scatter (where the error lives)\n"
   "`seed_cost` (x) vs `learned_value` (y), one point per perturbation. Points near x=0 mean the "
   "seed barely matters; points with y>0 mean learned propagation helps.")
code("plot_seed_vs_learned(res, SIGNIFICANT).savefig('fig_D_seed_vs_learned.png', dpi=140); None")

md("### Single-gene 2×2 (explainer, not a result)\n"
   "What one perturbation's 2×2 grid of E-distances looks like — useful to understand the layout, "
   "but the absolute values are background-dominated so do not read it as a result.")
code("plot_2x2(res['grids'][EVAL[0]], title=EVAL[0])")

md("## 7. Ranking + go/no-go note\n"
   "Lowest `end_to_end` E-distance = best, but remember it is background-dominated — prefer the "
   "skill leaderboard. The real go/no-go is the **rho_niche** (with vs without niche) **+0.10** "
   "gate, to be wired in once the no-niche variant exists. And `learned_value > 0` alone does "
   "**not** prove signal-specificity — confirm with figure B and a future label-permutation "
   "control.")
code("print('ranking (best first):', res['ranking'])")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "run_benchmark.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)
