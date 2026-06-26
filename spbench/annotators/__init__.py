"""Pluggable cell-type annotation layer.

The annotator is the single source of `data.cell_type`, which fully determines the
niche-composition line (niche.compute_niche_composition -> comp_l1 / Overlap). Swap the
annotator by name (the reserved modification point) and re-run for sensitivity analysis.

Registry mirrors spbench.metrics / spbench.adapters. Classes register themselves on import;
`get_annotator(name, **kwargs)` builds a FRESH instance (annotators are stateful after fit).
"""
import numpy as np
from .base import Annotator

_REGISTRY: dict[str, type] = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get_annotator(name: str, **kwargs) -> Annotator:
    return _REGISTRY[name](**kwargs)


def list_annotators() -> list:
    return sorted(_REGISTRY)


def apply_annotation(data, annotator: Annotator, context=None):
    """Fit `annotator` on the dataset's real cells, then relabel `data.cell_type` with it.
    Returns (data, fitted_annotator); the fitted instance is reused on predicted expression
    downstream so gt and pred share one frozen instrument. Mutates `data` in place."""
    ann = annotator.fit(data.X, data.cell_type, gene_names=data.gene_names, context=context)
    data.cell_type = np.asarray(ann.predict(data.X, gene_names=data.gene_names, context=context),
                                dtype=object)
    data.meta = dict(data.meta)
    data.meta["annotator"] = ann.name
    return data, ann


def annotation_fidelity(ann: Annotator, X, true_labels, gene_names=None) -> float:
    """QC: agreement between annotator(real expression) and expert labels. Report once per
    dataset (it measures the instrument, not any model); do NOT put on the model leaderboard."""
    pred = np.asarray(ann.predict(X, gene_names=gene_names))
    return float(np.mean(pred == np.asarray(true_labels, dtype=object)))


def annotate_from_config(data, cfg: dict):
    """Opt-in pipeline hook: relabel `data` per cfg['annotator'] (+ 'annotator_kwargs',
    'annotator_context'). No 'annotator' key -> returns (data, None) unchanged (back-compat)."""
    name = cfg.get("annotator")
    if not name:
        return data, None
    ann = get_annotator(name, **cfg.get("annotator_kwargs", {}))
    return apply_annotation(data, ann, context=cfg.get("annotator_context"))


from . import given, marker  # noqa: E402  (self-register)
