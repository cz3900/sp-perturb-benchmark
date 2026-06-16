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
    """Cell (2) = GT seed + learned propagation. If it is ~0 the learned model is likely
    copying the neighbour truth (leakage)."""
    return grid["2"]["energy_prop"] >= floor


def rho_niche_gate(rho_with: float, rho_without: float, margin: float = 0.10) -> bool:
    return (rho_with - rho_without) >= margin


def rank_models(end_to_end_scores: dict) -> list:
    """end_to_end_scores: {model_name: mean E-distance}. Lower is better."""
    return sorted(end_to_end_scores, key=lambda m: end_to_end_scores[m])
