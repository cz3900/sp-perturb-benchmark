"""scGEN as an offline, learned D1 SeedModel — a LOADER, mirroring ConcertModel.

scGEN trains in its own conda env (scvi-tools/jax/lightning pins conflict with the shared 3.11
venv), so it is run OFFLINE (scripts/scgen/run_scgen.py): export StandardData -> log-norm AnnData,
train one scGEN per perturbation, predict the perturbed state for the harness's per-center matched
controls, and dump the FINAL seed_pred array (n_centers x G), already aligned to centers order, to
`{P}_seed.h5ad`. This wrapper reads that array and serves it directly as the model seed — exactly
like ConcertModel serves a cached counterfactual.

Per the design's Part C fixes: predict_seed keeps the ABC's two-arg signature
(perturbation, reference_cells) and does NOT add a ref_idx third arg; the cached array IS the
prediction, so reference_cells is not used to re-derive it. The runner has already honored the
aggregate-control contract (it decodes the latent delta on the cell-type-mean control profile).
Constructed directly with prediction_paths (like ConcertModel) — NOT @register'd / auto-imported,
because __init__ requires prediction_paths.

Note: fit(train) is single-arg (SeedModel.fit, base.py:8), unlike ConcertModel.fit(train, edges).
"""
import numpy as np  # noqa: F401  (kept for back-compat imports)
from .seed_dump import SeedDumpModel, read_center_idx  # noqa: F401  (re-exported)
from .concert_model import read_h5ad_X  # noqa: F401  (re-exported, used by tests)


class ScgenSeedModel(SeedDumpModel):
    """scGEN's offline seed predictions, served via the generic SeedDumpModel contract."""

    def __init__(self, prediction_paths):
        super().__init__("scgen", prediction_paths)
