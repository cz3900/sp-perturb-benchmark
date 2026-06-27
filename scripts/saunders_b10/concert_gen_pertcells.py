import os, h5py, numpy as np
H = os.path.expanduser("~/spatial-pert/outputs/concert/saunders_b10.h5")
OUT = os.path.expanduser("~/spatial-pert/outputs/concert")
GENES = ["Hnf4a", "Ldlr", "Insr", "Srebf1", "B2m"]
with h5py.File(H, "r") as f:
    pert = np.array(f["perturbation"]).astype(str)
for P in GENES:
    idx = np.where(pert == P)[0] + 1          # 1-based, in exported-h5 cell order
    np.savetxt(f"{OUT}/pert_cells_{P}.txt", idx, fmt="%d")
    print(f"{P}: {len(idx)} cells -> pert_cells_{P}.txt")
