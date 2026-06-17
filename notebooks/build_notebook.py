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
   "*observed perturbed* neighbour distribution. Lower = better. `e4` (model seed + learned prop, "
   "aka `model+learned`) is the real, deployable score.\n"
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
   "**The one comparison that matters — beat the no-effect baseline.** The 2×2 only compares the "
   "four cells to *each other*; on its own it can't tell you whether *any* of them beats doing "
   "nothing. So we add one more prediction in the **same currency** (E-distance to the observed "
   "niche):\n"
   "- `e_null` = E-distance(**control niche**, observed niche) = the score of the laziest guess, "
   "'**the neighbours did not change**'. This is the bar to beat.\n"
   "- `oracle` = best a *non-leaking* model could reach (perfect mean shift + control-population "
   "variance) = a ceiling.\n"
   "\n"
   "Then the headline quantity is the **gain**:\n"
   "\n"
   "  `gain = e_null − e_method`  — **>0** the method beats 'no effect'; **<0** it is *worse* than "
   "doing nothing; bigger = better. No ratios, no clipping. `gain(model+learned)` is the bottom "
   "line (the deployable pipeline); `gain(oracle)` shows how much signal is recoverable at all.\n"
   "\n"
   "All of `e1..e4`, `e_null`, `oracle` are computed at a **matched sample size** (same n, same "
   "observed subsample per repeat) so the energy distance's finite-sample bias cancels and the "
   "gains are clean paired differences.\n"
   "\n"
   "**Two tasks, each with its own metric (decoupled).**\n"
   "- **seed** (the perturbed cell's own change) is scored *directly*: `seed_pcc` = **PCC-delta** "
   "= Pearson correlation between the predicted and true gene-wise shift (`perturbed − control`); "
   "`seed_mse` = magnitude of the mean error. PCC-delta is bounded [−1,1] and self-anchored (0 = "
   "no directional skill), so it sidesteps the energy distance's fragility.\n"
   "- **propagation** (the niche) is scored with the energy distance / `gain` (above) **plus** a "
   "niche **PCC-delta** (`niche_pcc`) — direction of the niche's gene-wise shift — as a robust "
   "cross-check. Sanity: `oracle`'s niche PCC-delta ≈ 1, the `null`'s is undefined (flat shift).")

md("## 1. Imports")
code("import matplotlib\n%matplotlib inline\n"
     "from spbench.adapters import get_adapter\n"
     "from spbench.config import run_benchmark\n"
     "from spbench.viz import (plot_2x2, plot_aggregate_2x2, plot_baseline_gain,\n"
     "                         plot_gain_per_perturbation, plot_learned_value,\n"
     "                         plot_significance_contrast)\n"
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
   "E-distance. `compare=True` also computes the no-effect baseline `e_null`, the `oracle` "
   "ceiling, and the per-method **gain = e_null − e** (see section 0). (`k` = neighbours per cell; "
   "`k_ref` = matched control cells per perturbed cell.)")
code("res = run_benchmark(data, perturbations=EVAL, k=15, k_ref=5,\n"
     "                    gcn_kwargs={'hidden':64,'epochs':30})")

md("## 5. Metric table\n"
   "One row per perturbation, organised by the two decoupled tasks. **seed**: `seed_pcc` "
   "(direction of the cell's own shift, −1..1, higher better), `seed_mse` (magnitude error, lower "
   "better). **propagation (deployable model+learned)**: `niche_gain = e_null − e` (**>0 beats "
   "'no effect'**), `niche_pcc` (direction of the niche shift). Context: `e_null` (baseline) and "
   "`gain_oracle` (recoverable ceiling). Sorted by `niche_gain` (best first).")
code("rows=[]\n"
     "for p in EVAL:\n"
     "    c=res['compare'][p]; s=res['seed'][p]\n"
     "    rows.append(dict(perturbation=p, sig=p in set(SIGNIFICANT),\n"
     "                     seed_pcc=s['pcc_delta'], seed_mse=s['mse'],\n"
     "                     niche_gain=c['gain']['model+learned'], niche_pcc=c['pcc']['model+learned'],\n"
     "                     e_null=c['e']['null'], gain_oracle=c['gain'].get('oracle'),\n"
     "                     leak_ok=res['leakage_pass'][p]))\n"
     "df=pd.DataFrame(rows).sort_values('niche_gain', ascending=False); df")

md("## 6. Result figures\n"
   "The headline is whether each method beats the no-effect baseline (`gain > 0`). The figures "
   "aggregate `gain` across perturbations and, for the deployable model, break it down per gene "
   "and contrast significant vs non-significant.")

md("### Headline — each method vs the no-effect baseline\n"
   "One box (+ one point per perturbation) per method, of **`gain = e_null − e`**. The solid line "
   "at **0 is the no-effect baseline**: a method is useful only where its points sit **above** it. "
   "The dashed line is the **oracle ceiling** (best a non-leaking model could reach). One glance: "
   "does any method beat doing nothing, by how much, and how far from the ceiling.")
code("plot_baseline_gain(res).savefig('fig_gain_aggregate.png', dpi=140); None")

md("### Summary 2×2 — mean gain over baseline\n"
   "All perturbations collapsed into one grid: each cell is the per-gene `gain = e_null − e` "
   "**averaged** across genes (so it is normalised to each gene's own baseline). **Green/>0 beats "
   "'no effect', red/<0 loses.** The **column difference** is the mean learned_value (learned vs "
   "Gaussian prop), the **row difference** is the mean seed_cost. Each cell also shows how many of "
   "the N perturbations individually beat the baseline — read this together with the spread in the "
   "headline box plot, since a single mean can hide a half-win/half-lose split.")
code("plot_aggregate_2x2(res).savefig('fig_aggregate_2x2.png', dpi=140); None")

md("### Per gene — deployable model vs baseline\n"
   "`gain = e_null − e[model+learned]` for every perturbation, sorted, coloured by significance. "
   "**Right of the line (>0) = the deployable pipeline predicts this gene's niche better than "
   "assuming 'no effect'.** Shows *which* genes (if any) it wins on, and whether they are the "
   "MC-significant ones.")
code("plot_gain_per_perturbation(res, SIGNIFICANT).savefig('fig_gain_per_pert.png', dpi=140); None")

md("### Diagnostic — is the learned advantage signal-specific?\n"
   "`learned_value = e1 − e2` (GCN vs Gaussian, holding the oracle seed) for significant vs "
   "non-significant groups, with a sign-test. **If the significant group is not clearly higher, "
   "the GCN's edge over Gaussian is generic** (a better smoother), not specific to real spatial "
   "signal.")
code("plot_significance_contrast(res, SIGNIFICANT).savefig('fig_signal_specificity.png', dpi=140); None")

md("### Single-gene 2×2 (explainer, not a result)\n"
   "What one perturbation's 2×2 grid of E-distances looks like — useful to understand the layout. "
   "Read it together with `e_null` (the baseline) for that gene, not in isolation.")
code("plot_2x2(res['grids'][EVAL[0]], title=EVAL[0])")

md("## 7. Bottom line\n"
   "Count how many perturbations the deployable pipeline actually beats the baseline on "
   "(`gain_deploy > 0`), and whether they are the MC-significant ones. `learned_value > 0` alone "
   "does **not** prove signal-specificity — confirm with the contrast figure and a future "
   "label-permutation control.")
code("g = {p: res['compare'][p]['gain']['model+learned'] for p in EVAL}\n"
     "wins = [p for p in EVAL if g[p] > 0]\n"
     "print(f'{len(wins)}/{len(EVAL)} perturbations beat the no-effect baseline:', wins)\n"
     "print('  of which MC-significant:', [p for p in wins if p in set(SIGNIFICANT)])")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "run_benchmark.ipynb")
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)
