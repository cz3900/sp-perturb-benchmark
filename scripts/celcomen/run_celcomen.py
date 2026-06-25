"""Offline Celcomen runner — dedicated python=3.9 env (torch/PyG). Two stages: (1) CCE learns the
gene-to-gene + intracellular regulation matrices; (2) Simcomen sets those matrices and applies the KO
via set_sphex (zero the guide gene's column), then generates the counterfactual spatial expression,
dumped to layers['counterfactual']. Off-panel guides have no column to zero -> skipped (no-op,
option 2). The pure helper below is unit-tested in the venv; main() needs the celcomen env."""
import argparse
import numpy as np


def ko_gene_index(guide_gene, genes):
    """Column index of the KO guide gene for Simcomen set_sphex zeroing. Returns None when the guide
    is OFF-PANEL (Celcomen can only KO an in-panel gene) — the runner then skips it."""
    genes = list(map(str, genes))
    return genes.index(str(guide_gene)) if str(guide_gene) in genes else None


def main():                                          # pragma: no cover (needs celcomen env)
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)         # from export_to_celcomen_h5 (custom h5 layout)
    ap.add_argument("--pert", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import h5py                                       # lazy
    import anndata as ad
    import pandas as pd
    from celcomen.models.celcomen import Celcomen
    from celcomen.models.simcomen import Simcomen

    # The adapter writes a CUSTOM HDF5 layout (not a valid .h5ad), so read it with raw h5py and build
    # the AnnData in memory — mirrors how the loader reads dumps.
    with h5py.File(args.h5ad, "r") as f:
        X = np.asarray(f["X"], float)
        cell_type = np.array(f["obs"]["cell_type"]).astype(str)
        batch = np.array(f["obs"]["batch"]).astype(str)
        spatial = np.asarray(f["obsm"]["spatial"], float)
        genes = [g.decode() if isinstance(g, bytes) else str(g)
                 for g in np.array(f["var"]["gene_names"])]
    j = ko_gene_index(args.pert, genes)
    if j is None:
        raise SystemExit(f"guide {args.pert!r} is off-panel; Celcomen cannot KO it")
    adata = ad.AnnData(X=X, obs=pd.DataFrame({"cell_type": cell_type, "batch": batch}),
                       var=pd.DataFrame({"gene_names": genes}, index=genes))
    adata.obsm["spatial"] = spatial
    # (1) CCE learns g2g + intracellular matrices  (2) Simcomen sets them, KOs gene j, generates.
    # Exact CCE/Simcomen signatures follow Tutorial_Celcomen_on_Xenium.ipynb / the spatial_KO tutorial;
    # verify + adjust on the server (Task 3.5) before the first real run.
    cce = Celcomen(input_dim=len(genes), output_dim=len(genes), n_neighbors=6)
    cce.fit(adata)
    sim = Simcomen(input_dim=len(genes), output_dim=len(genes), n_neighbors=6)
    sim.set_g2g(cce.get_g2g()); sim.set_g2g_intra(cce.get_g2g_intra())
    sphex = sim.x_to_sphex(np.asarray(adata.X, float))
    sphex[:, j] = 0.0                                # knock out guide gene
    sim.set_sphex(sphex)
    cf = np.asarray(sim.sphex_to_x(sim.generate()), float)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("X", data=np.zeros_like(cf))
        f.create_group("layers").create_dataset("counterfactual", data=cf)
    print(f"wrote {args.out}: counterfactual {cf.shape}")


if __name__ == "__main__":                           # pragma: no cover
    main()
