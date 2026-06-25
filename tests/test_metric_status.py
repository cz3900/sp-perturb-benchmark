from spbench.metrics import get_metric, list_metrics


def test_pcc_delta_is_active_primary():
    assert get_metric("pcc_delta").status == "active"
    assert get_metric("pcc_delta").higher_is_better is True


def test_comp_l1_and_moran_are_planned():
    assert get_metric("comp_l1").status == "planned"
    assert get_metric("moran_gap").status == "planned"


def test_list_metrics_active_only_excludes_planned():
    active = list_metrics(active_only=True)
    assert "pcc_delta" in active and "mse" in active and "energy" in active
    assert "comp_l1" not in active and "moran_gap" not in active
    # default still lists everything (back-compat)
    assert set(list_metrics()) >= set(active) | {"comp_l1", "moran_gap"}
