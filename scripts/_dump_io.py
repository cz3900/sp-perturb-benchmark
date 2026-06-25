"""Shared, conda-env-free dump IO for every offline seed runner. Pure numpy/h5py so the shared
venv can import it. Both functions are the contract SeedDumpModel reads back."""
import numpy as np, h5py


def aggregate_control_predict(center_cell_types, ct_profiles):
    """Map each center to its cell type's aggregate predicted profile.

    center_cell_types : (n_centers,) cell-type label per center (centers order).
    ct_profiles       : {cell_type: (G,) predicted perturbed profile}; key None = global fallback.
    Returns seed_pred (n_centers, G) aligned to centers order — the aggregate-control prediction."""
    center_cell_types = np.asarray(center_cell_types)
    glob = ct_profiles.get(None)
    G = len(next(iter(ct_profiles.values())))
    out = np.empty((len(center_cell_types), G), float)
    for i, ct in enumerate(center_cell_types):
        prof = ct_profiles.get(ct, glob)
        if prof is None:
            raise KeyError(f"no profile for cell type {ct!r} and no global (None) fallback")
        out[i] = prof
    return out


def write_seed_dump(path, seed_pred, center_idx):
    """Write the final aligned seed_pred + center indices for ScgenSeedModel to read.
    /X = (n_centers, G); /obs/center_idx = (n_centers,) StandardData center indices."""
    seed_pred = np.asarray(seed_pred, float)
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=seed_pred)
        g_obs = f.create_group("obs")
        g_obs.create_dataset("center_idx", data=np.asarray(center_idx, np.int64))
