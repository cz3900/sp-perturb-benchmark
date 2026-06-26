import numpy as np


def attribute(grid: dict) -> dict:
    """Read the 2x2 by differencing niche PCC-deltas (`pcc_prop`, higher = better)."""
    p = {k: grid[k]["pcc_prop"] for k in ("1", "2", "3", "4")}
    return {
        "seed_cost": p["1"] - p["3"],        # >0: model-seed worse than GT seed
        "learned_value": p["2"] - p["1"],    # >0: learned propagation beats baseline
        "end_to_end": p["4"],                # deployable score (higher = better)
    }


def leakage_pass(grid: dict, ceiling: float = 1.0 - 1e-3) -> bool:
    """A propagation prediction that reproduces the observed niche (PCC-delta ~1) indicates leakage.
    Check both GT-seed cells: (1) baseline+GT seed and (2) learned+GT seed. Either being ~1 means
    propagation started from the observed niche rather than a control reference. Missing cells
    default to -inf (pass), so a partial grid like {"2": ...} still works for unit tests."""
    p1 = grid.get("1", {}).get("pcc_prop", float("-inf"))
    p2 = grid.get("2", {}).get("pcc_prop", float("-inf"))
    return p1 <= ceiling and p2 <= ceiling
