import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
guides = ["Hnf4a","Ldlr","Insr","Srebf1","B2m","Dnajb9","Dut","Hspe1","Hyou1","Tap1","Xbp1"]
niche = {
 "Gaussian":   [-0.088,-0.861,-0.837,-0.446, 0.101, 0.005,-0.645,-0.624,-0.868,-0.663,-0.865],
 "GCN":        [-0.122,-0.813,-0.804,-0.475, 0.014, 0.191,-0.620,-0.643,-0.817,-0.642,-0.821],
 "SpatialProp":[ 0.517,-0.299,-0.201, 0.459, 0.214, 0.514, 0.440,-0.472,-0.731,-0.075,-0.566],
 "CONCERT":    [ 0.303, 0.833, 0.721, 0.524,-0.104, 0.096, 0.787, 0.612, 0.815, 0.772, 0.806]}
seed = {
 "TrivialSeed":[ 0.169, 0.507, 0.041,-0.011, 0.260, 0.392, 0.609, 0.259,-0.203, 0.666,-0.000],
 "SpatialProp":[ 0.612, 0.461, 0.856,-0.117, 0.668, 0.426, 0.630, 0.491,-0.016, 0.930, 0.357],
 "CONCERT":    [ 0.500,-0.260, 0.398, 0.673,-0.572, 0.290, 0.313,-0.174, 0.286,-0.550, 0.641]}
cols = guides + ["MEAN"]
def mat(d):
    rows = list(d); M = np.array([d[r] + [float(np.nanmean(d[r]))] for r in rows]); return rows, M
fig, axes = plt.subplots(2, 1, figsize=(15, 7.5), gridspec_kw={"height_ratios":[4,3]})
for ax, (title, d) in zip(axes, [("NICHE  PCC-delta  (bystander neighbours)", niche),
                                  ("SEED  PCC-delta  (perturbed centre cell)", seed)]):
    rows, M = mat(d)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(M[i,j])>0.6 else "black",
                    fontweight="bold" if j==len(guides) else "normal")
    ax.axvline(len(guides)-0.5, color="k", lw=1.5)
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
fig.suptitle("Saunders b10  ·  PCC-delta model benchmark (11 in-panel guides)   —   higher = better,  0 = no directional skill",
             fontsize=13, fontweight="bold")
fig.colorbar(im, ax=axes, shrink=0.55, label="PCC-delta", pad=0.015)
out = "/home/chengzheng/spatial-pert/outputs/figures/pcc_compare_b10.png"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)
