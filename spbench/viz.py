import numpy as np
import matplotlib.pyplot as plt

def plot_2x2(grid, title=""):
    """Heatmap of the 2x2 E-distance grid."""
    M = np.array([[grid["1"]["energy_prop"], grid["2"]["energy_prop"]],
                  [grid["3"]["energy_prop"], grid["4"]["energy_prop"]]])
    fig, ax = plt.subplots(figsize=(4, 3.2))
    im = ax.imshow(M, cmap="viridis_r")
    ax.set_xticks([0, 1], ["baseline prop", "learned prop"])
    ax.set_yticks([0, 1], ["GT seed", "model seed"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w")
    ax.set_title(f"2x2 E-distance {title}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig

def plot_attribution(attrib: dict):
    """Bar chart: per-perturbation end-to-end score + seed_cost + learned_value."""
    perts = list(attrib)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(perts))
    ax.bar(x - 0.25, [attrib[p]["end_to_end"] for p in perts], 0.25, label="end-to-end")
    ax.bar(x, [attrib[p]["seed_cost"] for p in perts], 0.25, label="seed cost")
    ax.bar(x + 0.25, [attrib[p]["learned_value"] for p in perts], 0.25, label="learned value")
    ax.set_xticks(x, perts)
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(); ax.set_title("2x2 attribution per perturbation")
    fig.tight_layout()
    return fig
