import numpy as np, pandas as pd, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN
from chhal.data import load_base_data
from chhal.detector import Detector
from chhal.evaluation import threshold_for_fpr
from chhal.fidelity import ks_by_vector, CONTROLLED_FEATURES
from chhal.optimizer import EvasionOptimizer
from chhal.redteam import ALL_VECTORS
from chhal.redteam.base import BaseProfile
from chhal.redteam.hosts import HostPool
t0=time.time()
base = load_base_data(source="ieee")
rng = np.random.default_rng(7)
prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
test_hosts = HostPool(base.test, exclude_accounts=base.train['_account'])
det = Detector(seed=7).fit(base.train, LABEL_COLUMN)
legit = base.test[base.test[LABEL_COLUMN]==0]
fraud = base.test[base.test[LABEL_COLUMN]==1]
lp = det.score(legit[FEATURE_COLUMNS])
thr = threshold_for_fpr(lp, 0.001)
print("realised fpr", float((lp>=thr).mean()))
print("REAL FRAUD recall@0.1%%FPR = %.4f   (n=%d)" % (float((det.score(fraud[FEATURE_COLUMNS])>=thr).mean()), len(fraud)))
opt = EvasionOptimizer(base.feature_stats)
rows=[]
for V in ALL_VECTORS:
    v = V().calibrate(prof, test_hosts)
    b = v.batch(500, 0, rng)
    raw_r = float((det.score(b.transactions[FEATURE_COLUMNS])>=thr).mean())
    ob = opt.optimize(b, det, rng)
    opt_r = float((det.score(ob.transactions[FEATURE_COLUMNS])>=thr).mean())
    ks_raw = ks_by_vector(legit, b.transactions, np.array([v.vector_id]*len(b)))
    ks_opt = ks_by_vector(legit, ob.transactions, np.array([v.vector_id]*len(ob)))
    rows.append(dict(vector=v.vector_id, raw_recall=raw_r, opt_recall=opt_r,
                     raw_ks_ctrl=float(ks_raw.mean_ks_controlled.iloc[0]),
                     opt_ks_ctrl=float(ks_opt.mean_ks_controlled.iloc[0])))
    print(rows[-1], flush=True)
print(pd.DataFrame(rows).to_string(index=False))
print("elapsed", time.time()-t0)
