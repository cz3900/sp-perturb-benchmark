import os, importlib.util, numpy as np
_R = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "spatialprop", "run_spatialprop.py")
def _load():
    spec = importlib.util.spec_from_file_location("run_spatialprop", _R)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def test_build_perturbation_dict_in_panel():
    m = _load()
    d = m.build_perturbation_dict("g2", ["T", "B"], ["g1", "g2", "g3"])
    assert d == {"T": {"g2": 0.0}, "B": {"g2": 0.0}}
def test_build_perturbation_dict_off_panel_none():
    m = _load()
    assert m.build_perturbation_dict("gX", ["T"], ["g1", "g2"]) is None
def test_two_group_split_has_two_disjoint_groups():
    m = _load()
    g = m.two_group_split(50)
    assert set(g) == {"g0", "g1"} and len(g) == 50
