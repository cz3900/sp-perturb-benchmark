# tests/test_annotators.py
import numpy as np
import pytest

from spbench.annotators import (
    Annotator, get_annotator, list_annotators,
    apply_annotation, annotation_fidelity, annotate_from_config,
)


def _separable(seed=0, per=30, n_genes=6):
    """3 cell types, each marked by a single high gene (A->g0, B->g1, C->g2).
    Cleanly separable, so a correct marker annotator must recover every label."""
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for ti, t in enumerate(["A", "B", "C"]):
        x = rng.normal(0.0, 0.1, size=(per, n_genes))
        x[:, ti] += 5.0
        blocks.append(x)
        labels += [t] * per
    X = np.vstack(blocks)
    y = np.array(labels, dtype=object)
    genes = [f"g{i}" for i in range(n_genes)]
    return X, y, genes


# --- registry -------------------------------------------------------------

def test_marker_and_given_registered():
    names = list_annotators()
    assert "marker" in names and "given" in names


def test_get_annotator_returns_fresh_instances():
    a = get_annotator("marker", n_markers=3)
    b = get_annotator("marker", n_markers=3)
    assert isinstance(a, Annotator) and a is not b      # stateful -> new instance each call


# --- given (passthrough) --------------------------------------------------

def test_given_is_identity_on_same_cells():
    X, y, genes = _separable()
    ann = get_annotator("given").fit(X, y, gene_names=genes)
    out = ann.predict(X, gene_names=genes)
    assert np.array_equal(out, y)


def test_given_refuses_to_infer_new_cells():
    X, y, genes = _separable()
    ann = get_annotator("given").fit(X, y, gene_names=genes)
    with pytest.raises(ValueError):
        ann.predict(X[:5], gene_names=genes)            # row-count mismatch -> cannot passthrough


# --- marker: data-driven dict (from expert labels) ------------------------

def test_marker_recovers_separable_labels():
    X, y, genes = _separable()
    ann = get_annotator("marker", n_markers=1).fit(X, y, gene_names=genes)
    pred = ann.predict(X, gene_names=genes)
    assert np.array_equal(pred, y)                      # fidelity == 1 on clean data


def test_marker_frozen_instrument_is_deterministic():
    X, y, genes = _separable()
    ann = get_annotator("marker", n_markers=1).fit(X, y, gene_names=genes)
    # same frozen stats applied to a shifted ("predicted") matrix -> same rule, repeatable
    Xpred = X + 0.3
    assert np.array_equal(ann.predict(Xpred, gene_names=genes),
                          ann.predict(Xpred, gene_names=genes))


# --- marker: external dict (no labels available) --------------------------

def test_marker_from_external_dict_without_labels():
    X, y, genes = _separable()
    md = {"A": ["g0"], "B": ["g1"], "C": ["g2"]}
    ann = get_annotator("marker").fit(X, ref_labels=None, gene_names=genes,
                                      context={"marker_dict": md})
    assert np.array_equal(ann.predict(X, gene_names=genes), y)


def test_marker_needs_labels_or_dict():
    X, y, genes = _separable()
    with pytest.raises(ValueError):
        get_annotator("marker").fit(X, ref_labels=None, gene_names=genes)


# --- fidelity QC ----------------------------------------------------------

def test_fidelity_is_one_on_separable():
    X, y, genes = _separable()
    ann = get_annotator("marker", n_markers=1).fit(X, y, gene_names=genes)
    assert annotation_fidelity(ann, X, y, gene_names=genes) == pytest.approx(1.0)


# --- pipeline integration -------------------------------------------------

def test_apply_annotation_sets_cell_type_and_meta(synth):
    ann = get_annotator("marker", n_markers=2)
    data, fitted = apply_annotation(synth, ann)
    assert data.cell_type.shape[0] == data.n_cells
    assert data.meta["annotator"] == "marker"
    assert set(np.unique(data.cell_type)).issubset(set(np.unique(synth.cell_type)))


def test_annotate_from_config_noop_without_key(synth):
    before = synth.cell_type.copy()
    data, ann = annotate_from_config(synth, {})
    assert ann is None
    assert np.array_equal(data.cell_type, before)       # back-compat: no 'annotator' -> unchanged


def test_annotate_from_config_applies_marker(synth):
    data, ann = annotate_from_config(
        synth, {"annotator": "marker", "annotator_kwargs": {"n_markers": 2}})
    assert ann is not None and ann.name == "marker"
    assert data.meta["annotator"] == "marker"


def test_run_from_yaml_applies_annotator(monkeypatch, tmp_path):
    import yaml
    from spbench import config
    from spbench.synthetic import make_synthetic

    data0 = make_synthetic(seed=0)

    class _FakeAdapter:
        def __init__(self, **kw):
            pass

        def load(self):
            return data0

    captured = {}
    monkeypatch.setattr(config, "get_adapter", lambda name: _FakeAdapter)
    monkeypatch.setattr(config, "run_benchmark",
                        lambda data, **kw: captured.setdefault("data", data) or {})

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({"adapter": "x", "perturbations": ["P0"],
                                 "annotator": "marker",
                                 "annotator_kwargs": {"n_markers": 2}}))
    config.run_from_yaml(str(p))
    assert captured["data"].meta["annotator"] == "marker"
