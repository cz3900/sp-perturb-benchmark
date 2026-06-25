# scripts/gears/run_gears.py
"""Offline GEARS runner — dedicated `gears` conda env (PyG + a pre-downloaded GO graph), NOT venv.

Per perturbation P: export `{P}.h5ad` (gears_export) + `{P}_centers.npz`; (gears env) train GEARS,
predict the perturbation's ONE mean expression vector, broadcast it to the n_centers, dump
`{P}_seed.h5ad`. GEARS predicts a single condition mean (not per cell), so every center of P gets
the same seed vector — broadcast_seed makes that explicit. Pure functions tested in the venv."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _dump_io import write_seed_dump  # noqa: E402


def broadcast_seed(pred_mean, n_centers):
    """GEARS' single per-perturbation mean (G,) -> (n_centers, G) aligned to centers order."""
    return np.tile(np.asarray(pred_mean, float), (int(n_centers), 1))


def main():                                              # pragma: no cover (needs the gears env)
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True); ap.add_argument("--centers", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--pert", required=True)
    args = ap.parse_args()
    from gears import GEARS, PertData           # lazy: only in the gears env
    # ... build PertData, train GEARS, pred_mean = model.predict([[args.pert]])[args.pert] ...
    npz = np.load(args.centers, allow_pickle=True)
    seed = broadcast_seed(pred_mean, len(npz["center_idx"]))    # noqa: F821
    write_seed_dump(args.out, seed, npz["center_idx"])


if __name__ == "__main__":                               # pragma: no cover
    main()
