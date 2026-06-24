import numpy as np
from types import SimpleNamespace
from spbench.adapters.cheng import _to_standarddata


def _cell(center, expr, top1, ctype, ds="b1"):
    return SimpleNamespace(CellCenter=np.array(center, float), MERFISHNum=np.array(expr, float),
                           Top1ID=top1, CellType=ctype, DataSet=ds)


def test_cheng_mapping():
    genes = ["g0", "g1", "g2"]
    targets = ["ABL1", "NTC", "TP53"]                  # Top1ID 1->ABL1, 2->NTC(control), 3->TP53
    cells = [
        _cell([1, 2], [1, 0, 0], 1, "Single"),          # -> ABL1
        _cell([3, 4], [0, 1, 0], 3, "Single"),          # -> TP53
        _cell([5, 6], [0, 0, 1], 2, "Single"),          # -> NTC target -> 'control'
        _cell([7, 8], [2, 2, 2], 99, "Non_decoded"),    # -> 'none'
        _cell([9, 0], [3, 3, 3], 1, "Dual"),            # -> 'none' (multiplet)
        _cell([0, 0], [1, 1], 1, "Single"),             # malformed expr (len 2 != 3) -> skipped
    ]
    data = _to_standarddata(genes, targets, cells, "MERFISHNum")
    assert list(data.gene_names) == genes
    assert data.n_cells == 5                            # malformed one dropped
    assert list(data.perturbation) == ["ABL1", "TP53", "control", "none", "none"]
    assert list(np.unique(data.cell_type)) == ["A549"]
    assert data.X.shape == (5, 3) and np.allclose(data.X[0], [1, 0, 0])
    assert data.coords.shape == (5, 2)
    assert data.has_ntc is True                         # the NTC->control cell gives a real control pool
    assert set(data.perturbations()) == {"ABL1", "TP53"}
