import numpy as np
from spbench.models.trivial_seed import TrivialSeed

def test_predicts_reference_plus_global_mean_shift(synth):
    m = TrivialSeed().fit(synth)
    refs = synth.X[synth.is_control][:10]
    pred = m.predict_seed("P0", refs)
    assert pred.shape == refs.shape
    pred2 = m.predict_seed("UNSEEN_GENE", refs)
    assert np.allclose(pred, pred2)
