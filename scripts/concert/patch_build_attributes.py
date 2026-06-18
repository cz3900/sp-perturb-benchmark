#!/usr/bin/env python
"""Patch CONCERT's run_concert_map.py so `build_attributes` is DATA-DRIVEN instead of hard-coded to
Perturb-Map biology.

CONCERT's original `build_attributes` derives tissue from a hard-coded gene whitelist
({Jak2,Tgfbr2,Ifngr2,KP}->tumor ...) and only codes those known perturbagens — so ANY other
dataset's perturbations (e.g. Saunders CRISPR KO genes) fall through to background=0 (fatal: every
perturbation treated as unperturbed). This patch replaces it with a generic version:
  * tissue codes  <- the stored `tissue` field (we export cell_type there)
  * perturbation codes <- unique non-background labels (background -> 0)
and threads the stored tissue through the caller.

Usage:  python patch_build_attributes.py /path/to/CONCERT/src/run_concert_map.py
Idempotent; writes a one-time .bak. Re-run safe.
"""
import sys, pathlib

GENERIC = '''def build_attributes(perturb_raw, tissue_raw=None, background=("None",)):
    """DATA-DRIVEN (patched for spbench): tissue codes from the stored tissue field (cell_type),
    perturbation codes from the unique non-background labels. Works for any dataset."""
    if tissue_raw is None:
        tissue_raw = np.full(len(perturb_raw), "all", dtype=object)
    t_uniq = sorted(set(str(t) for t in tissue_raw.tolist()))
    t_map = {t: i for i, t in enumerate(t_uniq)}
    tissue_idx = np.array([t_map[str(t)] for t in tissue_raw], dtype=int)
    bg = set(background)
    p_uniq = sorted(set(str(p) for p in perturb_raw.tolist()) - bg)
    pert_map = {p: i + 1 for i, p in enumerate(p_uniq)}
    perturb_idx = np.array([pert_map.get(str(p), 0) for p in perturb_raw], dtype=int)
    return tissue_idx, t_map, perturb_idx, pert_map
'''

CALLER_OLD = "X, pos_raw, _, perturb_raw = load_h5_dataset(cfg.data_file)"
CALLER_NEW = "X, pos_raw, tissue_raw, perturb_raw = load_h5_dataset(cfg.data_file)"
CALL_OLD = "build_attributes(perturb_raw)"
CALL_NEW = "build_attributes(perturb_raw, tissue_raw)"

# Counterfactual: let --target_cell_tissue "keep" hold each cell's own tissue (flip only the
# perturbation). Lets us flip a perturbation onto cells without forcing one tissue code.
TISSUE_OLD = ('        tissue_code = None\n'
              '        if isinstance(cfg.target_cell_tissue, str):')
TISSUE_NEW = ('        tissue_code = None\n'
              '        if cfg.target_cell_tissue == "keep":\n'
              '            tissue_code = None\n'
              '        elif isinstance(cfg.target_cell_tissue, str):')


def patch(path):
    p = pathlib.Path(path)
    src = p.read_text()
    if "DATA-DRIVEN (patched for spbench)" in src:
        print("already patched:", path); return
    p.with_suffix(p.suffix + ".bak").write_text(src)            # backup once

    # replace the build_attributes function: from its def to its return line (inclusive)
    start = src.index("def build_attributes(")
    end = src.index("return tissue_idx, tissue_dict, perturb_idx, pert_map", start)
    end = src.index("\n", end) + 1
    src = src[:start] + GENERIC + src[end:]

    assert CALLER_OLD in src, "caller line not found (CONCERT version changed?)"
    src = src.replace(CALLER_OLD, CALLER_NEW).replace(CALL_OLD, CALL_NEW)

    assert TISSUE_OLD in src, "eval tissue block not found (CONCERT version changed?)"
    src = src.replace(TISSUE_OLD, TISSUE_NEW)               # --target_cell_tissue "keep" support

    p.write_text(src)
    print("patched (data-driven build_attributes):", path)


if __name__ == "__main__":
    patch(sys.argv[1] if len(sys.argv) > 1 else "run_concert_map.py")
