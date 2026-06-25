import numpy as np


def attribute(grid: dict) -> dict:
    """Read the 2x2 by differencing E-distances."""
    e = {k: grid[k]["energy_prop"] for k in ("1", "2", "3", "4")}
    return {
        "seed_cost": e["3"] - e["1"],        # >0: model-seed worse than GT seed
        "learned_value": e["1"] - e["2"],    # >0: learned propagation beats baseline
        "end_to_end": e["4"],                # deployable score
    }


def leakage_pass(grid: dict, floor: float = 1e-3) -> bool:
    """A propagation prediction that reproduces the observed niche (energy ~0) indicates leakage.
    Check both GT-seed cells: (1) baseline+GT seed and (2) learned+GT seed. Either being ~0 means
    propagation started from the observed niche rather than a control reference. Missing cells
    default to inf (pass), so a partial grid like {"2": ...} still works for unit tests."""
    e1 = grid.get("1", {}).get("energy_prop", float("inf"))
    e2 = grid.get("2", {}).get("energy_prop", float("inf"))
    return e1 >= floor and e2 >= floor
