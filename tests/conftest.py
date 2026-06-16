import numpy as np
import pytest
from spbench.synthetic import make_synthetic

@pytest.fixture
def synth():
    return make_synthetic(seed=0)
