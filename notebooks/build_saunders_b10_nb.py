"""Build notebooks/saunders_b10_benchmark.ipynb — the Saunders b10 SpatialProp-vs-CONCERT
evaluation as a runnable notebook. Constructs the notebook JSON programmatically (like
build_notebook.py). Run on the server (spatialprop env) via nbconvert --execute. Mirrors
scripts/saunders_b10/score_b10_full.py + plot_pcc_b10.py."""
import json, os

cells = []
def md(t):   cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": t})

md("# Saunders b10 — SpatialProp vs CONCERT (PCC-delta + composition Overlap)\n"
   "\n"
   "End-to-end spatial perturbation benchmark on **Saunders 2025 MERFISH liver, slice "
   "`Batch_10_Slice_0`** (67512 cells × 209 genes, 9 cell types), 11 in-panel guides.\n"
   "\n"
   "Two **complementary** scoring currencies on the identical predicted niches:\n"
   "- **PCC-delta** — does the prediction move the right *genes* in the right direction "
   "(expression shift)? Scored in **seed** (perturbed centre cell) and **niche** (bystander "
   "neighbours) dimensions.\n"
   "- **composition Overlap** *(new)* — does the prediction reproduce the right *cell-type "
   "composition* of the niche? `Overlap = 1 - TV ∈ [0,1]`, read as 'what fraction of the niche "
   "cell-type makeup is predicted right'. This is the spatially-native readout PCC-delta is blind "
   "to (PCC-delta is permutation-invariant over cells — it ignores who sits next to whom).\n"
   "\n"
   "All vs `run_benchmark`'s built-in baselines (Gaussian / GCN naive propagation, TrivialSeed, the "
   "`null` no-change floor). The three expression spaces (observed raw counts / CONCERT raw / "
   "SpatialProp normalize_total) are unified by **per-cell `row_normalize`** so both metrics are "
   "scale-clean and the **one frozen cell annotator** (below) applies identically to every method. "
   "Run on the lab server, spatialprop env.")

code("%matplotlib inline\n"
     "import sys, os, numpy as np, h5py\n"
     "sys.path.insert(0, os.path.expanduser('~/spatial-pert/repo'))\n"
     "import matplotlib.pyplot as plt\n"
     "from spbench.adapters import SaundersAdapter\n"
     "from spbench.config import run_benchmark\n"
     "from spbench.external import row_normalize, score_external_seed\n"
     "from spbench.models.spatialprop_model import SpatialPropModel\n"
     "from spbench.models.concert_model import ConcertModel\n"
     "from spbench.annotators import get_annotator, apply_annotation, annotation_fidelity\n"
     "from spbench.graph import build_knn_graph\n"
     "from spbench.composition import niche_effect_board\n"
     "\n"
     "GUIDES = ['Hnf4a','Ldlr','Insr','Srebf1','B2m','Dnajb9','Dut','Hspe1','Hyou1','Tap1','Xbp1']\n"
     "SP  = os.path.expanduser('~/spatial-pert/outputs/spatialprop/dumps_Saunders_b10')\n"
     "CON = os.path.expanduser('~/spatial-pert/outputs/concert/out_b10')\n"
     "spaths = {P: f'{SP}/{P}.h5ad' for P in GUIDES}\n"
     "cpaths = {P: f'{CON}/saunders_b10_map_keep_{P}_perturbed_counts.h5ad' for P in GUIDES}")

code("def read_layer(p, L):\n"
     "    with h5py.File(p, 'r') as f: return np.asarray(f['layers'][L], float)\n"
     "def read_X(p):\n"
     "    with h5py.File(p, 'r') as f:\n"
     "        x = f['X']\n"
     "        if isinstance(x, h5py.Dataset): return np.asarray(x, float)\n"
     "        from scipy.sparse import csr_matrix\n"
     "        return csr_matrix((np.asarray(x['data']), np.asarray(x['indices']), np.asarray(x['indptr'])),\n"
     "                          shape=tuple(int(s) for s in x.attrs['shape'])).toarray().astype(float)\n"
     "\n"
     "class NormNiche:  # per-cell normalize external niche to data.X space (PCC scale-invariant)\n"
     "    def __init__(self, base): self.base = base\n"
     "    def fit(self, *a, **k): return self\n"
     "    def predict_niche(self, data, p, edges):\n"
     "        nb = self.base.predict_niche(data, p, edges); return row_normalize(nb) if len(nb) else nb\n"
     "    def predict(self, *a, **k): return self.base.predict(*a, **k)")

md("## Load Saunders b10\n"
   "`data.X` is set to `row_normalize(raw counts)`; SpatialProp/CONCERT niche predictions are wrapped "
   "with `NormNiche` so all three live in one normalize_total space.")

code("data = SaundersAdapter(os.path.expanduser('~/spatial-pert/inputs/saunders_b10_slice'),\n"
     "                       max_files=1, counts_layer='X').load()\n"
     "data.X = row_normalize(data.meta['counts'])\n"
     "print('cells', data.n_cells, '| genes', data.n_genes, '| cell types', len(set(data.cell_type)))")

md("## Cell annotation — the unified instrument\n"
   "\n"
   "The composition Overlap needs **cell-type labels** for the *observed* niche AND each model's "
   "*predicted* niche. The rule that keeps the comparison fair: **label both sides with ONE frozen "
   "annotator**, so the annotator's own systematic error cancels and Overlap reflects *model* error, "
   "not labeling error. (A model is never scored against the dataset's expert labels directly — that "
   "would conflate the annotator's error with the model's.)\n"
   "\n"
   "We use a **marker-score annotator** (reference-free, deterministic, pure-numpy):\n"
   "1. **Fit** on the observed cells — from the dataset's own cell-type labels, derive the top "
   "marker genes per type (a `rank_genes_groups`-style standardized mean difference) and freeze the "
   "per-gene mean/σ. No model training, no train/test leakage.\n"
   "2. **Predict** — z-score a cell with the frozen stats, score each cell-type's marker set, assign "
   "the arg-max type. The **same frozen instrument** is applied to observed and predicted expression.\n"
   "\n"
   "Why marker here (not CellTypist/scANVI): identical across datasets, deterministic, and robust on "
   "model-*predicted* (off-manifold) expression, where a learned classifier is unreliable. The expert "
   "labels are used to *build* the instrument, not to score models.\n"
   "\n"
   "**Fidelity** = agreement between `marker(observed expression)` and the dataset's expert labels — "
   "a one-off QC of the instrument (reported once; never on the model leaderboard). We then relabel "
   "`data.cell_type` with this frozen annotator so the descriptive board and the predictive Overlap "
   "share one labeling.")

code("ann = get_annotator('marker', n_markers=20).fit(data.X, data.cell_type, gene_names=data.gene_names)\n"
     "fid = annotation_fidelity(ann, data.X, data.cell_type, gene_names=data.gene_names)\n"
     "print('cell types :', list(ann.cats_))\n"
     "print(f'fidelity   : {fid:.3f}  (marker(observed) vs dataset expert labels; 1.0 = exact)')\n"
     "data, ann = apply_annotation(data, ann)   # relabel data.cell_type with the SAME frozen instrument\n"
     "print('annotator  :', data.meta['annotator'])")

md("## Run the benchmark\n"
   "`annotator=ann` is threaded through, so every method's predicted niche is labelled by the one "
   "frozen instrument and a composition **Overlap** is returned next to **PCC-delta** in "
   "`res['compare']`.")

code("res = run_benchmark(data, GUIDES, k=15, gcn_kwargs={'hidden':64,'epochs':30}, progress=False,\n"
     "                    annotator=ann,\n"
     "                    external_models={'SpatialProp': NormNiche(SpatialPropModel(spaths)),\n"
     "                                     'CONCERT': NormNiche(ConcertModel(cpaths))})")

md("## niche PCC-delta (bystander neighbours; higher = better)")
code("print(f\"{'guide':8} {'n':>4} | {'Gauss':>7} {'GCN':>7} | {'SpatProp':>8} {'CONCERT':>8}\")\n"
     "nA = {'g':[], 'gc':[], 'sp':[], 'con':[]}\n"
     "for P in GUIDES:\n"
     "    pcc = res['compare'][P]['pcc']; n = int((data.perturbation==P).sum())\n"
     "    g,gc,sp,con = (pcc.get(k, np.nan) for k in ('model+base','model+learned','SpatialProp','CONCERT'))\n"
     "    for k,v in zip(nA, (g,gc,sp,con)): nA[k].append(v)\n"
     "    print(f'{P:8} {n:4d} | {g:>7.3f} {gc:>7.3f} | {sp:>8.3f} {con:>8.3f}')\n"
     "print(f\"{'MEAN':8} {'':>4} | {np.nanmean(nA['g']):>7.3f} {np.nanmean(nA['gc']):>7.3f} | \"\n"
     "      f\"{np.nanmean(nA['sp']):>8.3f} {np.nanmean(nA['con']):>8.3f}\")")

md("## niche composition Overlap (cell-type makeup; higher = better)\n"
   "\n"
   "Same predicted niches as the PCC table, different question: instead of *expression direction*, "
   "**does the predicted niche have the right cell-type composition?** Each method's predicted "
   "bystander niche is labelled by the frozen marker instrument and its composition compared to the "
   "observed niche's via `Overlap = 1 - TV`. `null` is the predict-no-change baseline (the control "
   "niche) — **a method is only useful if it beats `null`**.")
code("print(f\"{'guide':8} {'n':>4} | {'Gauss':>7} {'GCN':>7} | {'SpatProp':>8} {'CONCERT':>8} | {'null':>6}\")\n"
     "oA = {'g':[], 'gc':[], 'sp':[], 'con':[], 'null':[]}\n"
     "for P in GUIDES:\n"
     "    ov = res['compare'][P]['overlap']; n = int((data.perturbation==P).sum())\n"
     "    g,gc,sp,con,nu = (ov.get(k, np.nan) for k in ('model+base','model+learned','SpatialProp','CONCERT','null'))\n"
     "    for k,v in zip(oA, (g,gc,sp,con,nu)): oA[k].append(v)\n"
     "    print(f'{P:8} {n:4d} | {g:>7.3f} {gc:>7.3f} | {sp:>8.3f} {con:>8.3f} | {nu:>6.3f}')\n"
     "print(f\"{'MEAN':8} {'':>4} | {np.nanmean(oA['g']):>7.3f} {np.nanmean(oA['gc']):>7.3f} | \"\n"
     "      f\"{np.nanmean(oA['sp']):>8.3f} {np.nanmean(oA['con']):>8.3f} | {np.nanmean(oA['null']):>6.3f}\")")

md("## seed PCC-delta (perturbed centre cell; higher = better)")
code("print(f\"{'guide':8} {'n':>4} | {'TrivSeed':>8} | {'SpatProp':>8} {'CONCERT':>8}\")\n"
     "sA = {'ts':[], 'sp':[], 'con':[]}\n"
     "for P in GUIDES:\n"
     "    ts = res['seed'][P]['pcc_delta']\n"
     "    sp = score_external_seed(data, P, row_normalize(read_layer(spaths[P],'predicted_tempered')))['pcc_delta']\n"
     "    con = score_external_seed(data, P, row_normalize(read_X(cpaths[P])))['pcc_delta']\n"
     "    n = int((data.perturbation==P).sum())\n"
     "    for k,v in zip(sA, (ts,sp,con)): sA[k].append(v)\n"
     "    print(f'{P:8} {n:4d} | {ts:>8.3f} | {sp:>8.3f} {con:>8.3f}')\n"
     "print(f\"{'MEAN':8} {'':>4} | {np.nanmean(sA['ts']):>8.3f} | \"\n"
     "      f\"{np.nanmean(sA['sp']):>8.3f} {np.nanmean(sA['con']):>8.3f}\")")

md("## Descriptive: which guides actually reshape the niche?\n"
   "\n"
   "Before asking models to *predict* the effect, quantify the **observed** effect per guide — the "
   "ground-truth / TARDIS-style readout, computed with our own Overlap on the unified labels. Two "
   "baselines:\n"
   "- **ovNTC** — KO-cell niche vs the control niche (raw effect vs unperturbed);\n"
   "- **ovPool** — KO-cell niche vs the *average perturbed* niche → the **guide-specific** shift, "
   "de-confounding any shared 'where do guide-bearing cells sit' axis.\n"
   "\n"
   "`p` = label-permutation test (shuffle KO vs control, keep n_ko; N=1000). Lower Overlap = bigger "
   "niche reshaping; rows sorted by the guide-specific shift, with the cell types that move most.")
code("edges = build_knn_graph(data, k=15)\n"
     "board = niche_effect_board(data, edges, min_ko=1, n_perm=1000, seed=0, perturbations=GUIDES)\n"
     "cats = board['cats']\n"
     "print(f\"{'guide':8}{'nKO':>5}{'ovNTC':>7}{'ovPool':>8}{'p':>7}   top guide-specific shifts (pooled delta %)\")\n"
     "for r in board['rows']:\n"
     "    d = np.asarray(r['delta_pooled']); idx = np.argsort(-np.abs(d))[:3]\n"
     "    sh = '  '.join(f'{cats[i]}({d[i]*100:+.1f})' for i in idx)\n"
     "    print(f\"{r['guide']:8}{r['n_ko']:5d}{r['overlap_ntc']:7.3f}{r['overlap_pooled']:8.3f}\"\n"
     "          f\"{r.get('p_value', float('nan')):7.3f}   {sh}\")")

md("## PCC-delta heatmap (niche + seed)")
code("def heat(ax, d, title):\n"
     "    rows = list(d); cols = GUIDES + ['MEAN']\n"
     "    M = np.array([d[r] + [float(np.nanmean(d[r]))] for r in rows])\n"
     "    im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')\n"
     "    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha='right', fontsize=9)\n"
     "    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=10)\n"
     "    for i in range(M.shape[0]):\n"
     "        for j in range(M.shape[1]):\n"
     "            ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center', fontsize=7.5,\n"
     "                    color='white' if abs(M[i,j])>0.6 else 'black', fontweight='bold' if j==len(GUIDES) else 'normal')\n"
     "    ax.axvline(len(GUIDES)-0.5, color='k', lw=1.5); ax.set_title(title, fontsize=12, loc='left', fontweight='bold')\n"
     "    return im\n"
     "niche_d = {'Gaussian':nA['g'],'GCN':nA['gc'],'SpatialProp':nA['sp'],'CONCERT':nA['con']}\n"
     "seed_d  = {'TrivialSeed':sA['ts'],'SpatialProp':sA['sp'],'CONCERT':sA['con']}\n"
     "fig, axes = plt.subplots(2, 1, figsize=(15, 7.5), gridspec_kw={'height_ratios':[4,3]})\n"
     "heat(axes[0], niche_d, 'NICHE  PCC-delta  (bystander neighbours)')\n"
     "im = heat(axes[1], seed_d, 'SEED  PCC-delta  (perturbed centre cell)')\n"
     "fig.suptitle('Saunders b10 · PCC-delta benchmark (11 in-panel guides) — higher = better, 0 = no skill',\n"
     "             fontsize=13, fontweight='bold')\n"
     "fig.colorbar(im, ax=axes, shrink=0.55, label='PCC-delta', pad=0.015)\n"
     "out = os.path.expanduser('~/spatial-pert/outputs/figures/pcc_compare_b10.png')\n"
     "os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, dpi=140, bbox_inches='tight')\n"
     "plt.show(); print('saved', out)")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(__file__), "saunders_b10_benchmark.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
