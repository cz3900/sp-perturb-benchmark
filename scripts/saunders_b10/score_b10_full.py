import sys, os, numpy as np, h5py
sys.path.insert(0, os.path.expanduser("~/spatial-pert/repo"))
from spbench.adapters import SaundersAdapter
from spbench.config import run_benchmark
from spbench.external import row_normalize, score_external_seed
from spbench.models.spatialprop_model import SpatialPropModel
from spbench.models.concert_model import ConcertModel

def read_layer(p, L):
    with h5py.File(p, "r") as f: return np.asarray(f["layers"][L], float)
def read_X(p):
    with h5py.File(p, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Dataset): return np.asarray(x, float)
        from scipy.sparse import csr_matrix
        return csr_matrix((np.asarray(x["data"]), np.asarray(x["indices"]), np.asarray(x["indptr"])),
                          shape=tuple(int(s) for s in x.attrs["shape"])).toarray().astype(float)

class NormNiche:
    def __init__(self, base): self.base = base
    def fit(self, *a, **k): return self
    def predict_niche(self, data, p, edges):
        nb = self.base.predict_niche(data, p, edges); return row_normalize(nb) if len(nb) else nb
    def predict(self, *a, **k): return self.base.predict(*a, **k)

GUIDES = ["Hnf4a","Ldlr","Insr","Srebf1","B2m","Dnajb9","Dut","Hspe1","Hyou1","Tap1","Xbp1"]
SP = os.path.expanduser("~/spatial-pert/outputs/spatialprop/dumps_Saunders_b10")
CON = os.path.expanduser("~/spatial-pert/outputs/concert/out_b10")
spaths = {P: f"{SP}/{P}.h5ad" for P in GUIDES}
cpaths = {P: f"{CON}/saunders_b10_map_keep_{P}_perturbed_counts.h5ad" for P in GUIDES}
data = SaundersAdapter(os.path.expanduser("~/spatial-pert/inputs/saunders_b10_slice"),
                       max_files=1, counts_layer="X").load()
data.X = row_normalize(data.meta["counts"])
res = run_benchmark(data, GUIDES, k=15, gcn_kwargs={"hidden":64,"epochs":30},
                    progress=False, external_models={"SpatialProp": NormNiche(SpatialPropModel(spaths)),
                                                      "CONCERT": NormNiche(ConcertModel(cpaths))})
print("\n================ niche PCC-delta (higher=better) ================")
print(f"{'guide':8} {'n':>4} | {'Gauss':>7} {'GCN':>7} | {'SpatProp':>8} {'CONCERT':>8}")
nA={'sp':[],'con':[],'g':[],'gc':[]}
for P in GUIDES:
    pcc = res["compare"][P]["pcc"]; n=int((data.perturbation==P).sum())
    g,gc,sp,con = pcc.get('model+base',np.nan),pcc.get('model+learned',np.nan),pcc.get('SpatialProp',np.nan),pcc.get('CONCERT',np.nan)
    nA['g'].append(g);nA['gc'].append(gc);nA['sp'].append(sp);nA['con'].append(con)
    print(f"{P:8} {n:4d} | {g:>7.3f} {gc:>7.3f} | {sp:>8.3f} {con:>8.3f}")
print(f"{'MEAN':8} {'':>4} | {np.nanmean(nA['g']):>7.3f} {np.nanmean(nA['gc']):>7.3f} | {np.nanmean(nA['sp']):>8.3f} {np.nanmean(nA['con']):>8.3f}")
print("\n================ seed PCC-delta (higher=better) ================")
print(f"{'guide':8} {'n':>4} | {'TrivSeed':>8} | {'SpatProp':>8} {'CONCERT':>8}")
sA={'sp':[],'con':[],'ts':[]}
for P in GUIDES:
    ts = res["seed"][P]["pcc_delta"]
    sp = score_external_seed(data, P, row_normalize(read_layer(spaths[P],"predicted_tempered")))["pcc_delta"]
    con = score_external_seed(data, P, row_normalize(read_X(cpaths[P])))["pcc_delta"]
    n=int((data.perturbation==P).sum())
    sA['ts'].append(ts);sA['sp'].append(sp);sA['con'].append(con)
    print(f"{P:8} {n:4d} | {ts:>8.3f} | {sp:>8.3f} {con:>8.3f}")
print(f"{'MEAN':8} {'':>4} | {np.nanmean(sA['ts']):>8.3f} | {np.nanmean(sA['sp']):>8.3f} {np.nanmean(sA['con']):>8.3f}")
