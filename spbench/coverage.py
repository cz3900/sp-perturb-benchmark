"""Split a dataset's guide genes into those a model can inject vs cannot, given the model's
injectable-gene set (panel genes for SpatialProp/Celcomen; TF∪ligand∪receptor for SpaceTravLR).
Off-panel guides are NOT dropped from the benchmark (they keep real GT marker shifts) — this only
tells the runner which guides a gene-column injector can represent (option 2)."""


def guide_overlap(guides, allowed):
    """guides, allowed: iterables of gene symbols. Returns {'in': sorted injectable guides,
    'out': sorted non-injectable guides}. Exact case-sensitive match (caller normalizes case)."""
    allowed = set(map(str, allowed))
    g = [str(x) for x in guides]
    return {"in": sorted([x for x in g if x in allowed]),
            "out": sorted([x for x in g if x not in allowed])}
