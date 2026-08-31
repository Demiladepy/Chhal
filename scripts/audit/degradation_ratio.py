import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scipy.stats import ks_2samp
from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN
from chhal.data import load_base_data
from chhal.fidelity import CONTROLLED_FEATURES
from chhal.redteam import ALL_VECTORS
from chhal.redteam.base import BaseProfile
from chhal.redteam.hosts import HostPool
base = load_base_data(source="ieee"); rng = np.random.default_rng(11)
fraud = base.test[base.test[LABEL_COLUMN]==1]; legit = base.test[base.test[LABEL_COLUMN]==0]
N=500; R=100
# Split legit ONCE into a reference and a disjoint probe pool. Every DR numerator is
# ks(legit_ref, population); the null floor in the denominator must be ks(legit_ref,
# independent legit) drawn from the DISJOINT probe pool. The earlier code took the floor
# as ks(full legit, subset OF full legit), so the null sample overlapped its own reference
# and the floor came out ~1.4x too low, inflating every DR by the same factor. R: 20 -> 100.
_gi = rng.permutation(len(legit)); _h = len(_gi)//2
legit_ref, legit_probe = legit.iloc[_gi[:_h]], legit.iloc[_gi[_h:]]
def mks(s): return float(np.mean([ks_2samp(legit_ref[c].to_numpy(), s[c].to_numpy()).statistic for c in CONTROLLED_FEATURES]))
def mks_n(pop, n=N, r=R):
    v=[mks(pop.sample(n=min(n,len(pop)), random_state=int(rng.integers(1e6)))) for _ in range(r)]
    return float(np.mean(v)), float(np.std(v))
floor, floor_sd = mks_n(legit_probe)
print(f"NOISE FLOOR (legit vs legit, n={N}, {R} draws) = {floor:.4f} (sd {floor_sd:.4f})\n")
print(f"{'population':38s} {'n':>5s} {'mean_ks@n=500':>14s} {'DR (=ks/floor)':>15s}")
rows=[("REAL: all test fraud",fraud),
      ("REAL: high-amount q80+",fraud[fraud.amount>=fraud.amount.quantile(.8)]),
      ("REAL: channel_code==1",fraud[fraud.channel_code==1]),
      ("REAL: new payee only",fraud[fraud.is_new_beneficiary==1]),
      ("REAL: night 0-5",fraud[fraud.hour<6]),
      ("REAL: top-q velocity_24h",fraud[fraud.velocity_24h>=fraud.velocity_24h.quantile(.8)]),
      ("REAL: burst vel+gap",fraud[(fraud.velocity_24h>=fraud.velocity_24h.quantile(.8))&(fraud.time_since_last_txn_min<=fraud.time_since_last_txn_min.quantile(.2))])]
for nm,p in rows:
    if len(p)<100: continue
    m,s = mks_n(p); print(f"{nm:38s} {len(p):5d} {m:14.4f} {m/floor:14.1f}x")
prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
th = HostPool(base.test, exclude_accounts=base.train['_account'])
print()
for V in ALL_VECTORS:
    v=V().calibrate(prof,th); b=v.batch(N,0,rng).transactions
    m=mks(b); print(f"{'CHHAL: '+v.vector_id+' (raw)':38s} {len(b):5d} {m:14.4f} {m/floor:14.1f}x")
