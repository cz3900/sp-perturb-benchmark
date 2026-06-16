import numpy as np
from spbench.split import split_by_perturbation

def test_unseen_perturbations_held_out(synth):
    sp = split_by_perturbation(synth, unseen=["P2"], seed=0)
    train_perts = set(np.unique(sp["train"].perturbation))
    assert "P2" not in train_perts
    assert "P2" in set(np.unique(sp["test"].perturbation))

def test_no_guide_cells_stay_in_train(synth):
    sp = split_by_perturbation(synth, unseen=["P2"], seed=0)
    assert sp["train"].is_unlabeled.sum() > 0
    assert sp["train"].is_control.sum() > 0
