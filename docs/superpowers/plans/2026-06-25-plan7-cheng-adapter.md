# Plan 7 — ChengAdapter (Perturb-RAEFISH A549, .mat + codebook) Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** `ChengAdapter(directory)` reads Cheng Perturb-RAEFISH (A549) `.mat` files into a `StandardData`, single cell line (degenerate mode, depends on Plan 2). Verified server structure (node03):
- `CellList_PerturbRaeFISH.mat::CellList_All` — array of 33141 cell structs; fields used: `MERFISHNum` (492-gene expression vector), `CellCenter` ([x,y]), `Top1ID` (1-based guide index), `CellType` (`Single`/`Dual`/`Non_decoded` decode status — NOT a biological cell type), `DataSet` (batch).
- `Codebook_MERFISH.mat::CodeBook` — 492 structs; `.ShortName` → gene names (X columns, in order).
- `Codebook_RaeFISH.mat::CodeBook` — 1001 structs; `.Target` → the guide's target gene (e.g. `ABL1`). `Top1ID` indexes this 1-based.
- **Mapping**: `cell_type = 'A549'` (constant; single line → Plan 2 degenerate). perturbation: `Single` → `Target[Top1ID-1]` (the KO gene), unless that target is a control name (NTC/non-targeting/safe-harbor → `'control'`); `Non_decoded`/`Dual` → `'none'` (no confident single guide → Plan 2's `control_pool` falls back to these). `X = MERFISHNum`, `coords = CellCenter[:2]`, `batch = DataSet`, `gene_names = MERFISH ShortName`.

**Architecture:** Factor the field→StandardData logic into a pure `_to_standarddata(genes, targets, cells, expr_field)` so the test drives it with plain `SimpleNamespace` cell records (no `.mat` round-trip). `_structs(path, key)` reads a mat struct-array via `scipy.io.loadmat(squeeze_me=True, struct_as_record=False)`. Real `.mat` I/O is verified on the server (controller-run); the unit test covers the mapping logic.

**Tech Stack:** Python 3.11, numpy, scipy.io, pytest.

---

## File Structure
- `spbench/adapters/cheng.py` (new): `ChengAdapter`, `_to_standarddata`, `_structs`.
- `spbench/adapters/__init__.py` (modify): register `"cheng"`.
- `tests/test_adapter_cheng.py` (new): pure-function mapping test (SimpleNamespace records).

---

### Task 1: ChengAdapter mapping logic + test

- [ ] **Step 1 — failing test** (`tests/test_adapter_cheng.py`):
```python
import numpy as np
from types import SimpleNamespace
from spbench.adapters.cheng import _to_standarddata

def _cell(center, expr, top1, ctype, ds="b1"):
    return SimpleNamespace(CellCenter=np.array(center, float), MERFISHNum=np.array(expr, float),
                           Top1ID=top1, CellType=ctype, DataSet=ds)

def test_cheng_mapping():
    genes = ["g0", "g1", "g2"]
    targets = ["ABL1", "NTC", "TP53"]           # Top1ID 1->ABL1, 2->NTC(control), 3->TP53
    cells = [
        _cell([1, 2], [1, 0, 0], 1, "Single"),      # -> ABL1
        _cell([3, 4], [0, 1, 0], 3, "Single"),      # -> TP53
        _cell([5, 6], [0, 0, 1], 2, "Single"),      # -> NTC target -> 'control'
        _cell([7, 8], [2, 2, 2], 99, "Non_decoded"),# -> 'none'
        _cell([9, 0], [3, 3, 3], 1, "Dual"),        # -> 'none' (multiplet)
        _cell([0, 0], [1, 1], 1, "Single"),         # malformed expr (len 2 != 3) -> skipped
    ]
    data = _to_standarddata(genes, targets, cells, "MERFISHNum")
    assert list(data.gene_names) == genes
    assert data.n_cells == 5                       # malformed one dropped
    assert list(data.perturbation) == ["ABL1", "TP53", "control", "none", "none"]
    assert list(np.unique(data.cell_type)) == ["A549"]
    assert data.X.shape == (5, 3) and np.allclose(data.X[0], [1, 0, 0])
    assert data.coords.shape == (5, 2)
    assert data.has_ntc is True                    # the NTC->control cell gives a real control pool
    assert set(data.perturbations()) == {"ABL1", "TP53"}
```

- [ ] **Step 2 — confirm FAIL** (`_to_standarddata` undefined).

- [ ] **Step 3 — implement `spbench/adapters/cheng.py`:**
```python
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
```

- [ ] **Step 4 — register** in `spbench/adapters/__init__.py`: `from .cheng import ChengAdapter` + `"cheng": ChengAdapter`.

- [ ] **Step 5 — PASS + full suite + commit.** `python -m pytest tests/test_adapter_cheng.py -q` PASS; `python -m pytest -q` green. `git add -A && git commit -m "feat(adapter): ChengAdapter (Perturb-RAEFISH A549 .mat + codebook -> StandardData, single line) (Plan 7)"`

---

## Self-Review
- **Coverage**: roadmap Plan 7 — Top1ID→Target gene (codebook), gene-name recovery (ShortName), single cell_type='A549' (Plan 2 degenerate), Single/Non_decoded/Dual → KO/none. Registry updated. Test asserts each mapping + malformed-drop + NTC→control.
- **Depends on Plan 2**: the 'none' cells become the control pool via `control_pool` (verified in Plan 2). When NTC-target guides exist they give an explicit 'control' (test asserts `has_ntc`).
- **Server verification (controller, after merge)**: `ChengAdapter('/home/yiru/.../Perturb_RAEFISH').load()` on node03 → check 33141 cells / 492 genes / real KO names; run_benchmark on one KO + plot.
- **Out of scope**: NormalizedMERFISHNum vs MERFISHNum (expr_field param defaults to raw counts), the RAEFISH whole-transcriptome version (separate data), Cheng's `Top2ID`/ratios.
