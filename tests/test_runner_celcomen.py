import os, importlib.util
_R = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                  "scripts", "celcomen", "run_celcomen.py")

def _load():
    spec = importlib.util.spec_from_file_location("run_celcomen", _R)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_ko_gene_index_in_panel():
    m = _load()
    assert m.ko_gene_index("g2", ["g1", "g2", "g3"]) == 1

def test_ko_gene_index_off_panel_none():
    m = _load()
    assert m.ko_gene_index("gX", ["g1", "g2"]) is None
