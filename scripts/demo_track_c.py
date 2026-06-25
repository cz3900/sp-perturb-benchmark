"""Track C smoke demo — three new seed models (CPA / GEARS / biolord) scored under ONE currency.

Two-stage flow (mirrors scripts/scgen):

  STAGE 1 (server, per-model conda env — NOT the shared .venv):
    Each of cpa / gears / biolord trains in its OWN offline env. See:
        scripts/cpa/ENV_SETUP.md      + scripts/cpa/run_cpa.py
        scripts/gears/ENV_SETUP.md    + scripts/gears/run_gears.py
        scripts/biolord/ENV_SETUP.md  + scripts/biolord/run_biolord.py
    Each runner trains one model per perturbation P and dumps `{P}_seed.h5ad`
    (/X aligned to centers order + /obs/center_idx) — the shared dump contract that
    SeedDumpModel / ScgenSeedModel read. No per-model loader needed.

  STAGE 2 (shared .venv — THIS script):
    Wire each dump into SeedDumpModel(name, {P: path}), run the real
    fill_2x2(...) -> evaluate_seed(...) seed-board path, and print a summary_table-shaped
    per-guide PCC-delta. PCC-delta (mean-shift direction + magnitude) is the single locked
    scoring currency: every model — Gaussian/GCN baselines, scGEN, and these three — lands
    on the same number, so the boards are directly comparable.

Why not run_benchmark for the seed models? run_benchmark builds its own TrivialSeed internally
and does not accept a custom seed model, so the per-model fill_2x2 / evaluate_seed loop (the same
path the integration guard test exercises) is the correct illustration of the seed board. The
end-to-end / niche externals go through run_benchmark(external_models=...); the seed dumps do not.

Run on synthetic (self-contained, no real dumps needed):
    python scripts/demo_track_c.py --synthetic

Run on real dumps (server, after STAGE 1 produced them):
    python scripts/demo_track_c.py --dump-dir ~/cpa_dumps/<dataset> --model cpa
"""
import argparse
import glob
import os

import numpy as np

from spbench.synthetic import make_synthetic
from spbench.graph import build_knn_graph
from spbench.harness import fill_2x2
from spbench.compare import evaluate_seed
from spbench.models.seed_dump import SeedDumpModel
from spbench.models.gaussian_prop import GaussianProp

MODELS = ("cpa", "gears", "biolord")


def score_seed_model(data, edges, base, name, prediction_paths):
    """Score one seed model on every perturbation it has a dump for.
    Returns {P: evaluate_seed(...) result} — the seed board for this model."""
    board = {}
    for P, path in prediction_paths.items():
        model = SeedDumpModel(name, {P: path}).fit(None)
        grid = fill_2x2(data, P, edges, model, base, base, return_niches=True)
        board[P] = evaluate_seed(grid["_niches"])
    return board


def summary_rows(boards):
    """One row per (model, guide): the single PCC-delta currency + MSE + matched n.
    `boards` is {model_name: {P: evaluate_seed_result}}. Returns list[dict]."""
    rows = []
    for name in sorted(boards):
        for P in sorted(boards[name]):
            r = boards[name][P]
            rows.append({"model": name, "guide": P,
                         "seed_pcc_delta": r["pcc_delta"],
                         "seed_mse": r["mse"], "n": r["n"]})
    return rows


def print_summary(rows):
    """Print the summary_table-style per-guide PCC-delta board (the locked currency)."""
    hdr = f"{'model':<10}{'guide':<12}{'pcc_delta':>12}{'mse':>12}{'n':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:<10}{r['guide']:<12}{r['seed_pcc_delta']:>12.4f}"
              f"{r['seed_mse']:>12.4f}{r['n']:>6}")


def _mock_dump(path, data, P):
    """Stand in for a STAGE-1 runner dump so the demo is runnable with no real model env:
    a finite seed shift on the perturbed centers, aligned to centers order (the dump contract)."""
    import h5py
    centers = np.where(data.perturbation == P)[0]
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=data.X[centers] + 0.2)
        f.create_group("obs").create_dataset("center_idx", data=centers.astype(np.int64))
    return path


def run_synthetic(tmp_dir):
    """Self-contained illustration: synthetic data + mock dumps for all three models."""
    data = make_synthetic(0)
    edges = build_knn_graph(data, k=8)
    base = GaussianProp().fit(data, edges)
    perturbations = data.perturbations()
    boards = {}
    for name in MODELS:
        paths = {}
        for P in perturbations:
            dump = os.path.join(tmp_dir, f"{name}_{P}_seed.h5ad")
            paths[P] = _mock_dump(dump, data, P)
        boards[name] = score_seed_model(data, edges, base, name, paths)
    return boards


def _discover_dumps(dump_dir):  # pragma: no cover  (needs real STAGE-1 output)
    """Map perturbation -> `{P}_seed.h5ad` path from a real STAGE-1 dump directory."""
    paths = {}
    for f in sorted(glob.glob(os.path.join(dump_dir, "*_seed.h5ad"))):
        P = os.path.basename(f)[: -len("_seed.h5ad")]
        paths[P] = f
    return paths


def run_real(dump_dir, model, dataset):  # pragma: no cover  (server-only, needs real dumps + adapter)
    """Sketch of the real adapter-loaded path. Runs only on the server where the real dumps and
    `/home/yiru/...` datasets exist; the import-safe code above is exercised by --synthetic / tests."""
    from spbench.adapters import get_adapter
    data = get_adapter(dataset).load()
    edges = build_knn_graph(data, k=15)
    base = GaussianProp().fit(data, edges)
    paths = _discover_dumps(dump_dir)
    return {model: score_seed_model(data, edges, base, model, paths)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true",
                    help="run on the built-in synthetic generator with mock dumps (default)")
    ap.add_argument("--dump-dir", help="server: directory of real `{P}_seed.h5ad` dumps")
    ap.add_argument("--model", choices=MODELS, help="server: which model the dumps belong to")
    ap.add_argument("--dataset", help="server: adapter key for the real dataset")
    args = ap.parse_args()

    if args.dump_dir:  # pragma: no cover  (server-only)
        if not (args.model and args.dataset):
            ap.error("--dump-dir requires --model and --dataset")
        boards = run_real(args.dump_dir, args.model, args.dataset)
    else:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            boards = run_synthetic(tmp)

    print_summary(summary_rows(boards))


if __name__ == "__main__":
    main()
