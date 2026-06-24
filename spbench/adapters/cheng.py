import numpy as np
from .base import DatasetAdapter
from ..data import StandardData

# guide targets that mean "not a real KO" -> control
_CONTROL_TARGET_TOKENS = ("ntc", "non-target", "nontarget", "non_target", "control",
                          "scramble", "safe", "negative")


def _is_control_target(t):
    s = str(t).lower()
    return any(tok in s for tok in _CONTROL_TARGET_TOKENS)


def _to_standarddata(genes, targets, cells, expr_field="MERFISHNum", name="Cheng_2025"):
    """Map parsed Cheng records -> StandardData (single A549 line). cells: iterable of records with
    attrs CellCenter, <expr_field>, Top1ID, CellType, DataSet. Single -> Target KO (or 'control' if
    the target is a non-targeting/control guide); Non_decoded/Dual -> 'none'. Drops cells whose
    expression length != len(genes)."""
    genes = list(genes)
    X, coords, pert, batch = [], [], [], []
    for c in cells:
        expr = np.asarray(getattr(c, expr_field), float).ravel()
        if expr.size != len(genes):
            continue
        ctype = str(getattr(c, "CellType"))
        if ctype == "Single":
            tid = int(getattr(c, "Top1ID"))
            tgt = targets[tid - 1] if 1 <= tid <= len(targets) else None
            if tgt is None:
                p = "none"
            elif _is_control_target(tgt):
                p = "control"
            else:
                p = str(tgt)
        else:
            p = "none"
        X.append(expr)
        cc = np.asarray(getattr(c, "CellCenter"), float).ravel()
        coords.append(cc[:2])
        pert.append(p)
        batch.append(str(getattr(c, "DataSet")))
    n = len(X)
    return StandardData(
        X=np.array(X, float) if n else np.zeros((0, len(genes))),
        coords=np.array(coords, float) if n else np.zeros((0, 2)),
        perturbation=np.array(pert), cell_type=np.full(n, "A549"),
        batch=np.array(batch), gene_names=genes, meta={"name": name},
    )


def _structs(path, key):
    import scipy.io as sio
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return list(np.ravel(m[key]))


class ChengAdapter(DatasetAdapter):
    """Cheng Perturb-RAEFISH (A549) .mat -> StandardData (single cell line, Plan-2 degenerate).
    MERFISHNum (492-gene expr), CellCenter (coords), Top1ID (1-based guide), CellType decode status,
    DataSet (batch); Codebook_MERFISH.ShortName = genes; Codebook_RaeFISH.Target = guide KO gene."""

    def __init__(self, directory, expr_field="MERFISHNum"):
        self.directory = directory
        self.expr_field = expr_field

    def load(self):
        d = self.directory
        genes = [str(c.ShortName) for c in _structs(d + "/Codebook_MERFISH.mat", "CodeBook")]
        targets = [str(c.Target) for c in _structs(d + "/Codebook_RaeFISH.mat", "CodeBook")]
        cells = _structs(d + "/CellList_PerturbRaeFISH.mat", "CellList_All")
        return _to_standarddata(genes, targets, cells, self.expr_field)
