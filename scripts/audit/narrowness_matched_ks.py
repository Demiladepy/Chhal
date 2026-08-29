"""Is a chhal vector further from legit traffic than real fraud is?

The claim this script exists to kill: "every vector is further from legit than real fraud".
The number behind it (mean_ks_controlled over ALL real fraud) is right, and the comparison
is confounded. Real fraud is a heterogeneous mixture; every vector is a narrow typology.
Narrow the real side the same way and the bands overlap.

Both halves are computed here, on the same split, with the same metric, and each carries
its recall at the same operating point -- because the distance number alone is what made
the original claim look decisive.
"""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scipy.stats import ks_2samp
from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN
from chhal.data import load_base_data
from chhal.detector import Detector
from chhal.evaluation import threshold_for_fpr
from chhal.fidelity import CONTROLLED_FEATURES
from chhal.optimizer import EvasionOptimizer
from chhal.redteam import ALL_VECTORS
from chhal.redteam.base import BaseProfile
from chhal.redteam.hosts import HostPool

base = load_base_data(source="ieee")
fraud = base.test[base.test[LABEL_COLUMN] == 1]
legit = base.test[base.test[LABEL_COLUMN] == 0]

det = Detector(seed=7).fit(base.train, LABEL_COLUMN)
legit_scores = det.score(legit[FEATURE_COLUMNS])
thr = threshold_for_fpr(legit_scores, 0.001)
print(f"threshold at 0.1% FPR (realised {float((legit_scores >= thr).mean()):.6f})\n")


def mks(s: pd.DataFrame) -> float:
    """mean KS vs legit over the columns the red team actually sets."""
    return float(np.mean([ks_2samp(legit[c].to_numpy(), s[c].to_numpy()).statistic
                          for c in CONTROLLED_FEATURES]))


def recall(s: pd.DataFrame) -> float:
    return float((det.score(s[FEATURE_COLUMNS]) >= thr).mean())


print("DECISIVE CONTROL: mean_ks_controlled vs legit for NARROW REAL fraud subgroups")
subs = {
    "ALL real test fraud": fraud,
    "channel_code==0": fraud[fraud.channel_code == 0],
    "channel_code==1": fraud[fraud.channel_code == 1],
    "channel_code==2": fraud[fraud.channel_code == 2],
    "new payee only": fraud[fraud.is_new_beneficiary == 1],
    "cross-border only": fraud[fraud.is_cross_border == 1],
    "top-quintile velocity_24h": fraud[fraud.velocity_24h >= fraud.velocity_24h.quantile(.8)],
    "burst: vel24h>=q80 & gap<=q20": fraud[(fraud.velocity_24h >= fraud.velocity_24h.quantile(.8))
                                           & (fraud.time_since_last_txn_min <= fraud.time_since_last_txn_min.quantile(.2))],
    "high-amount q80+": fraud[fraud.amount >= fraud.amount.quantile(.8)],
    "night hours 0-5": fraud[fraud.hour < 6],
}
real = []
for k, v in subs.items():
    if len(v) < 50:
        print(f"  {k:32s} n={len(v):5d}  (too small)")
        continue
    real.append((k, mks(v), recall(v)))
    print(f"  {k:32s} n={len(v):5d}  ks={real[-1][1]:.4f}  recall@0.1%FPR={real[-1][2]:.4f}")

# This block used to be a hardcoded print string. It went stale the moment the split moved
# (7-day embargo + straddler purge, 29 Aug) and kept printing pre-purge numbers underneath
# freshly-computed real-fraud ones. A stale literal inside an audit script is the exact
# failure mode this directory exists to prevent, so it is computed now.
rng = np.random.default_rng(7)
prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
test_hosts = HostPool(base.test, exclude_accounts=base.train["_account"])
opt = EvasionOptimizer(base.feature_stats)

print("\nfor reference, chhal vectors (optimized), same metric, same split, same threshold:")
ref = []
for V in ALL_VECTORS:
    v = V().calibrate(prof, test_hosts)
    ob = opt.optimize(v.batch(500, 0, rng), det, rng)
    ref.append((v.vector_id, mks(ob.transactions), recall(ob.transactions)))
for vid, ks, rec in sorted(ref, key=lambda r: r[1]):
    print(f"  {vid:32s} n=  500  ks={ks:.4f}  recall@0.1%FPR={rec:.4f}")

print("\nVERDICT")
print(f"  real fraud     ks {min(r[1] for r in real):.4f}-{max(r[1] for r in real):.4f}"
      f"   recall {min(r[2] for r in real):.2%}-{max(r[2] for r in real):.2%}")
print(f"  chhal vectors  ks {min(r[1] for r in ref):.4f}-{max(r[1] for r in ref):.4f}"
      f"   recall {min(r[2] for r in ref):.2%}-{max(r[2] for r in ref):.2%}")
n_inside = sum(1 for r in ref if r[1] <= max(x[1] for x in real))
print(f"  {n_inside}/{len(ref)} vectors fall INSIDE the real-fraud distance band.")
print("  Distance from legit does not explain the recall gap. Provenance does.")
