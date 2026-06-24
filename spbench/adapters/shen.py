import glob, os
import numpy as np, h5py
from .base import DatasetAdapter
from ..data import StandardData
from .saunders import _read_cat, _read_matrix

_SHEN_NONE_TOKENS = ("doublet", "non-perturb", "nonperturb", "undecoded")
_SHEN_CONTROL_TOKENS = ("safe", "control", "ntc", "non-target", "nontarget", "scramble")


def _map_shen_perturbation(values):
    """Map Shen's obs/perturbation labels to the StandardData convention. Gene names -> KO (kept);
    Doublet / Non-perturbed -> 'none' (multiplet / undecoded background; become the control pool via
    StandardData.control_pool, Plan 2); msafe / safe-harbor / NTC -> 'control'."""
    out = []
    for v in values:
        s = str(v); sl = s.lower()
        if any(t in sl for t in _SHEN_NONE_TOKENS):
            out.append("none")
        elif any(t in sl for t in _SHEN_CONTROL_TOKENS):
            out.append("control")
        else:
            out.append(s)
    return np.array(out)


class ShenAdapter(DatasetAdapter):
    """Shen spatial perturb-seq (Stereo-seq brain, whole-transcriptome ~21590), single-cell h5mu.
    mod/rna/X (sparse counts), obs/perturbation, obs/x + obs/y coords, var/_index genes. No
    cell_type column -> a single constant type (Plan-2 degenerate; n small, ~165 perturbed cells ->
    report n). Doublet/Non-perturbed -> 'none', msafe/safe -> 'control', gene -> KO (see
    `_map_shen_perturbation`)."""

    def __init__(self, directory, max_files=None, cell_type="brain"):
        self.directory = directory
        self.max_files = max_files
        self.cell_type = cell_type

    def load(self):
        files = sorted(glob.glob(os.path.join(self.directory, "*.h5mu")))
        if self.max_files:
            files = files[: self.max_files]
        Xs, coords, pert, batch, genes = [], [], [], [], None
        for f in files:
            with h5py.File(f, "r") as h:
                rna = h["mod/rna"]
                Xs.append(_read_matrix(rna["X"]))
                coords.append(np.column_stack([rna["obs/x"][:], rna["obs/y"][:]]).astype(float))
                pert.append(_map_shen_perturbation(_read_cat(rna["obs/perturbation"])))
                batch.append(np.full(Xs[-1].shape[0], os.path.basename(f)))
                if genes is None:
                    genes = [g.decode() if isinstance(g, bytes) else g for g in rna["var/_index"][:]]
        n = int(sum(x.shape[0] for x in Xs))
        return StandardData(
            X=np.vstack(Xs), coords=np.vstack(coords),
            perturbation=np.concatenate(pert).astype(str),
            cell_type=np.full(n, self.cell_type),
            batch=np.concatenate(batch).astype(str),
            gene_names=genes, meta={"name": "Shen_2024"},
        )
