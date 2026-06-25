import os, importlib.util
_R = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                  "scripts", "spacetravlr", "run_spacetravlr.py")

def _load():
    spec = importlib.util.spec_from_file_location("run_spacetravlr", _R)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_is_perturbable_tf_or_lr():
    m = _load()
    assert m.is_perturbable("Gata3", ["Gata3", "Il2"]) is True
    assert m.is_perturbable("Xyz", ["Gata3", "Il2"]) is False
