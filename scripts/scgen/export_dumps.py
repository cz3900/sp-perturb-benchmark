"""Step-1 export driver for the scGEN dump contract (runs in the SHARED env, e.g. concert).

Loads a Saunders slice, builds the log-norm X once, and for each perturbation writes the two
inputs run_scgen.py (scgen env) consumes:
  {P}.h5ad         via export_to_scgen_h5 (control vs P, log-norm X)
  {P}_centers.npz  'center_idx' (StandardData centers = perturbed cells of P) + 'cell_type',
                   both in centers order, so the seed dump aligns to what fill_2x2 scores.

A perturbation with no stimulated cells in this slice is skipped with a log line.
"""
import argparse
import os
import numpy as np

from spbench.adapters.saunders import SaundersAdapter
from spbench.adapters.scgen_export import build_lognorm_X, export_to_scgen_h5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", required=True, help="Saunders directory (.h5mu slices)")
    ap.add_argument("--out", required=True, help="output dir for {P}.h5ad + {P}_centers.npz")
    ap.add_argument("--max-files", type=int, default=1)
    ap.add_argument("--counts-layer", default="X")
    ap.add_argument("--perturbations", nargs="+", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = SaundersAdapter(args.slice_dir, max_files=args.max_files,
                           counts_layer=args.counts_layer).load()
    lognorm_X = build_lognorm_X(data)
    print(f"loaded: {data.n_cells} cells x {data.n_genes} genes; exporting to {args.out}")
    for p in args.perturbations:
        centers = np.where(data.perturbation == p)[0]
        if len(centers) == 0:
            print(f"skip {p}: no stimulated cells in this slice")
            continue
        info = export_to_scgen_h5(data, p, lognorm_X, os.path.join(args.out, f"{p}.h5ad"))
        np.savez(os.path.join(args.out, f"{p}_centers.npz"),
                 center_idx=centers.astype(np.int64),
                 cell_type=np.asarray([str(c) for c in data.cell_type[centers]]))
        print(f"{p}: stim={info['n_stim']} ctrl={info['n_ctrl']} centers={len(centers)}")
    print("export done.")


if __name__ == "__main__":
    main()
