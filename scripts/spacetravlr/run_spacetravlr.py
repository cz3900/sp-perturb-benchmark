"""Offline SpaceTravLR runner — dedicated uv env (celloracle/commot/torch + GPU). We BYPASS
SpaceShip.spawn_worker (which self-submits a Slurm job) and drive the model directly inside our own
srun allocation: setup_ -> fit -> setup_perturbations -> perturb. KO = perturb(target, gene_expr=0).
The perturbed matrix is aligned onto the exported panel via align_prediction_to_panel (genes
SpaceTravLR doesn't model are left at the unperturbed input) and dumped to
layers['predicted_perturbed']. Off-panel / off-GRN targets are skipped (no-op, option 2). The pure
helper below is unit-tested in the venv; main() needs the real env."""
import argparse
import numpy as np


def is_perturbable(target, perturbable):
    """SpaceTravLR can only perturb a TF (base GRN) or ligand/receptor (CellChat). `perturbable` is
    that allowed-gene set. Returns True iff `target` is injectable; else the runner skips it."""
    return str(target) in set(map(str, perturbable))


def main():                                          # pragma: no cover (needs spacetravlr env + GPU)
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)         # from export_to_spacetravlr_h5
    ap.add_argument("--pert", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    import h5py                                       # lazy
    import scanpy as sc
    from SpaceTravLR.spaceship import SpaceShip
    from spbench.dump_align import align_prediction_to_panel

    adata = sc.read_h5ad(args.h5ad)
    species = adata.uns.get("species", "mouse") if hasattr(adata, "uns") else "mouse"
    panel = list(adata.var["gene_names"].astype(str))
    ship = SpaceShip(name="bench").setup_(adata, run_commot=True)          # GRN + L-R preprocessing
    perturbable = set(SpaceShip.load_base_GRN(species).iloc[:, 0].astype(str))
    if not is_perturbable(args.pert, perturbable):
        raise SystemExit(f"target {args.pert!r} is not a TF/ligand/receptor SpaceTravLR can perturb")
    ship.fit(epochs=args.epochs)                                          # = run_spacetravlr (no sbatch)
    ship.setup_perturbations(adata)
    pred = np.asarray(ship.perturb(target=args.pert, gene_expr=0, propagation=4), float)
    pred_genes = list(ship.factory.genes) if hasattr(ship, "factory") else panel
    fallback = np.asarray(adata.X, float)
    aligned = align_prediction_to_panel(pred, pred_genes, panel, fallback)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("X", data=np.zeros_like(aligned))
        f.create_group("layers").create_dataset("predicted_perturbed", data=aligned)
    print(f"wrote {args.out}: predicted_perturbed {aligned.shape}")


if __name__ == "__main__":                           # pragma: no cover
    main()
