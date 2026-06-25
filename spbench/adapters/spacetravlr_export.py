"""StandardData -> SpaceTravLR input. SpaceTravLR (CellOracle/SpaceOracle descendant) needs an
AnnData with RAW counts in X (it builds a base-GRN + CellChat ligand-receptor graph internally in
setup_), obs['cell_type'], obs['batch'], obsm['spatial'], and a species tag that selects the
mouse/human base GRN. SINGLE-CELL resolution only — do NOT export Visium-spot data (Dhainaut)."""
import numpy as np, h5py


def export_to_spacetravlr_h5(data, counts_X, path, species="mouse"):
    counts_X = np.asarray(counts_X, float)
    if counts_X.shape != (data.n_cells, data.n_genes):
        raise ValueError(f"counts_X shape {counts_X.shape} != ({data.n_cells}, {data.n_genes})")
    if species not in ("mouse", "human"):
        raise ValueError(f"species must be 'mouse' or 'human', got {species!r}")
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=counts_X)
        g = f.create_group("obs")
        g.create_dataset("cell_type", data=np.asarray([str(c) for c in data.cell_type], dtype="S"))
        g.create_dataset("batch", data=np.asarray([str(b) for b in data.batch], dtype="S"))
        f.create_group("obsm").create_dataset("spatial", data=np.asarray(data.coords, float))
        f.create_group("var").create_dataset("gene_names",
                        data=np.asarray([str(x) for x in data.gene_names], dtype="S"))
        f.attrs["species"] = species
    return {"n_cells": int(data.n_cells), "species": species}
