import sys, os, numpy as np
sys.path.insert(0, os.path.expanduser("~/spatial-pert/repo"))
from spbench.adapters import SaundersAdapter
from spbench.adapters.concert_export import export_to_concert_h5
work = os.path.expanduser("~/spatial-pert/inputs/saunders_b10_slice")
out = os.path.expanduser("~/spatial-pert/outputs/concert/saunders_b10.h5")
os.makedirs(os.path.dirname(out), exist_ok=True)
data = SaundersAdapter(work, max_files=1, counts_layer="X").load()
counts = data.meta["counts"]
gene_keep = (counts > 0).sum(0) >= 10
cell_keep = counts[:, gene_keep].sum(1) >= 1
print("loaded", data.n_cells, "x", data.n_genes, "| genes<10det dropped",
      int((~gene_keep).sum()), "| cells dropped", int((~cell_keep).sum()))
data = data.subset(cell_keep); data.X = data.X[:, gene_keep]
data.gene_names = list(np.asarray(data.gene_names)[gene_keep])
counts_f = counts[cell_keep][:, gene_keep]
print("after filter:", data.n_cells, "x", data.n_genes, "|", len(data.perturbations()), "perturbations")
export_to_concert_h5(data, out, counts=counts_f)
print("exported ->", out)
for g in ["Hnf4a","Ldlr","Insr","Srebf1","B2m"]:
    print(" ", g, "n_cells", int((data.perturbation==g).sum()), "in_panel", g in data.gene_names)
