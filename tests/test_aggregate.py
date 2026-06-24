import numpy as np
from spbench.aggregate import normalized_score, cross_dataset_rank, rank_from_results


def test_normalized_score_floor_ceiling():
    # 0 at the null floor, 1 at the oracle ceiling, mid in between
    assert normalized_score(e=3.0, e_null=3.0, e_oracle=1.0) == 0.0
    assert normalized_score(e=1.0, e_null=3.0, e_oracle=1.0) == 1.0
    assert abs(normalized_score(e=2.0, e_null=3.0, e_oracle=1.0) - 0.5) < 1e-9
    # clipped to [0,1]
    assert normalized_score(e=0.5, e_null=3.0, e_oracle=1.0) == 1.0   # better than oracle -> 1
    assert normalized_score(e=4.0, e_null=3.0, e_oracle=1.0) == 0.0   # worse than null -> 0
    # degenerate gap -> nan
    assert np.isnan(normalized_score(e=2.0, e_null=2.0, e_oracle=2.0))


def test_cross_dataset_rank_ranks_and_aggregate():
    # absolute e differs wildly across datasets (like Visium ~400 vs MERFISH <1) but ranks align
    per = {
        "visium": {"A": {"e": 100.0, "null": 400.0, "oracle": 50.0},   # best (rank 1)
                   "B": {"e": 300.0, "null": 400.0, "oracle": 50.0}},  # worse (rank 2)
        "merfish": {"A": {"e": 0.20, "null": 0.90, "oracle": 0.10},     # best (rank 1)
                    "B": {"e": 0.60, "null": 0.90, "oracle": 0.10}},   # worse (rank 2)
    }
    out = cross_dataset_rank(per)
    assert out["ranks"]["visium"] == {"A": 1, "B": 2}
    assert out["ranks"]["merfish"] == {"A": 1, "B": 2}
    # A is rank 1 in both -> mean_rank 1.0; B -> 2.0
    assert out["aggregate"]["A"]["mean_rank"] == 1.0
    assert out["aggregate"]["B"]["mean_rank"] == 2.0
    assert out["aggregate"]["A"]["n_datasets"] == 2
    # A normalized higher than B in both datasets (despite absolute-e incomparability)
    assert out["normalized"]["visium"]["A"] > out["normalized"]["visium"]["B"]
    assert out["aggregate"]["A"]["mean_norm"] > out["aggregate"]["B"]["mean_norm"]


def test_rank_from_results_builds_per_dataset():
    # minimal fake run_benchmark res: res['compare'][p]['e'][method]
    def _res(eb, el):
        return {"compare": {"P0": {"e": {"model+base": eb, "model+learned": el,
                                          "null": 1.0, "oracle": 0.1}}}}
    out = rank_from_results({"d1": _res(0.5, 0.3), "d2": _res(0.4, 0.6)},
                            methods=("model+base", "model+learned"))
    # d1: learned(0.3) < base(0.5) -> learned rank 1; d2: base(0.4) < learned(0.6) -> base rank 1
    assert out["ranks"]["d1"]["model+learned"] == 1
    assert out["ranks"]["d2"]["model+base"] == 1
    assert set(out["aggregate"]) == {"model+base", "model+learned"}
    assert out["aggregate"]["model+base"]["n_datasets"] == 2
