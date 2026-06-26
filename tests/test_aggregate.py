import numpy as np
from spbench.aggregate import normalized_pcc, cross_dataset_rank, rank_from_results


def test_normalized_pcc_null_upper():
    # 0 at the null level, 1 at the GT-seed upper, mid in between
    assert normalized_pcc(pcc=0.0, pcc_null=0.0, pcc_upper=1.0) == 0.0
    assert normalized_pcc(pcc=1.0, pcc_null=0.0, pcc_upper=1.0) == 1.0
    assert abs(normalized_pcc(pcc=0.5, pcc_null=0.0, pcc_upper=1.0) - 0.5) < 1e-9
    # clipped to [0,1]
    assert normalized_pcc(pcc=1.2, pcc_null=0.0, pcc_upper=1.0) == 1.0   # better than upper -> 1
    assert normalized_pcc(pcc=-0.2, pcc_null=0.0, pcc_upper=1.0) == 0.0  # below null -> 0
    # degenerate / non-positive gap -> nan
    assert np.isnan(normalized_pcc(pcc=0.5, pcc_null=1.0, pcc_upper=1.0))


def test_cross_dataset_rank_ranks_and_aggregate():
    # absolute pcc differs across datasets/spaces but ranks align (higher pcc = better)
    per = {
        "visium": {"A": {"pcc": 0.80, "null": 0.0, "upper": 0.95},   # best (rank 1)
                   "B": {"pcc": 0.40, "null": 0.0, "upper": 0.95}},  # worse (rank 2)
        "merfish": {"A": {"pcc": 0.60, "null": 0.0, "upper": 0.90},   # best (rank 1)
                    "B": {"pcc": 0.20, "null": 0.0, "upper": 0.90}},  # worse (rank 2)
    }
    out = cross_dataset_rank(per)
    assert out["ranks"]["visium"] == {"A": 1, "B": 2}
    assert out["ranks"]["merfish"] == {"A": 1, "B": 2}
    # A is rank 1 in both -> mean_rank 1.0; B -> 2.0
    assert out["aggregate"]["A"]["mean_rank"] == 1.0
    assert out["aggregate"]["B"]["mean_rank"] == 2.0
    assert out["aggregate"]["A"]["n_datasets"] == 2
    # A normalized higher than B in both datasets (despite absolute-pcc incomparability)
    assert out["normalized"]["visium"]["A"] > out["normalized"]["visium"]["B"]
    assert out["aggregate"]["A"]["mean_norm"] > out["aggregate"]["B"]["mean_norm"]


def test_rank_from_results_builds_per_dataset():
    # minimal fake run_benchmark res: res['compare'][p]['pcc'][method]
    def _res(pb, pl):
        return {"compare": {"P0": {"pcc": {"model+base": pb, "model+learned": pl,
                                           "null": 0.0, "GT+learned": 0.95}}}}
    out = rank_from_results({"d1": _res(0.5, 0.7), "d2": _res(0.6, 0.4)},
                            methods=("model+base", "model+learned"))
    # d1: learned(0.7) > base(0.5) -> learned rank 1; d2: base(0.6) > learned(0.4) -> base rank 1
    assert out["ranks"]["d1"]["model+learned"] == 1
    assert out["ranks"]["d2"]["model+base"] == 1
    assert set(out["aggregate"]) == {"model+base", "model+learned"}
    assert out["aggregate"]["model+base"]["n_datasets"] == 2
