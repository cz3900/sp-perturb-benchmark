"""Export a StandardData (one slice) to the CONCERT input `.h5` format.

CONCERT (Lin et al., bioRxiv 2025) consumes an HDF5 with:
  /X            (N, G)  raw INTEGER counts (stored as float)
  /pos          (2, N)  spatial coords (transposed; CONCERT min-max scales them)
  /perturbation (N,)    byte strings   — per-cell perturbation label
  /tissue       (N,)    byte strings   — per-cell context (we use cell_type)
  /gene         (G,)    byte strings   — gene names

This is the reverse of an adapter: our common StandardData -> CONCERT's format, so CONCERT can
be trained/evaluated on our datasets (Saunders MERFISH etc.) through spbench. Pair this with the
generic, data-driven `build_attributes` patch (concert_repro/build_attributes_generic.py) so
CONCERT maps our arbitrary cell-type / perturbation labels instead of its hard-coded Perturb-Map
biology.
"""
import numpy as np
import h5py
from ..data import CONTROL, UNLABELED

BACKGROUND = "None"   # CONCERT's no-perturbation label (gets perturbation code 0)


def export_to_concert_h5(data, path, counts=None, background=BACKGROUND):
    """Write `data` (subset to a single slice) as a CONCERT `.h5` at `path`.

    counts : (N, G) raw integer counts. Defaults to data.X — but CONCERT asserts integer counts,
             so if data.X holds normalised/scaled values pass the raw layer here explicitly.
    Control / no-guide cells (`CONTROL`, `UNLABELED`) are written as `background` so CONCERT treats
    them as unperturbed. Returns the per-perturbation cell counts (for a sanity log)."""
    X = np.asarray(data.X if counts is None else counts, float)
    if X.shape[0] != data.n_cells:
        raise ValueError(f"counts has {X.shape[0]} rows but data has {data.n_cells} cells")
    if not np.allclose(X, np.round(X), atol=1e-6):
        raise ValueError("CONCERT requires RAW INTEGER counts, but X is non-integer "
                         "(data.X is probably normalised/scaled). Pass counts=<raw count layer>.")
    X = np.round(X).astype(np.float64)

    pert = np.array([background if p in (CONTROL, UNLABELED) else str(p)
                     for p in data.perturbation], dtype=object)
    pos = np.round(np.asarray(data.coords, float)).astype(np.int32).T          # (2, N)

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("pos", data=pos)
        f.create_dataset("perturbation", data=np.asarray(pert, dtype="S"))
        f.create_dataset("tissue", data=np.asarray([str(c) for c in data.cell_type], dtype="S"))
        f.create_dataset("gene", data=np.asarray([str(g) for g in data.gene_names], dtype="S"))

    uniq, cnt = np.unique(pert, return_counts=True)
    return dict(zip(uniq.tolist(), cnt.tolist()))
