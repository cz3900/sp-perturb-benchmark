import numpy as np
from spbench.reference import match_reference_centers

def test_matches_same_cell_type(synth):
    centers = np.where(synth.perturbation == "P0")[0]
    matched = match_reference_centers(synth, centers, k=3)
    assert len(matched) == len(centers)
    for c, refs in zip(centers, matched):
        assert len(refs) > 0
        assert np.all(synth.is_control[refs])
        assert np.all(synth.cell_type[refs] == synth.cell_type[c])
