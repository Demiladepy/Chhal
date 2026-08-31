import numpy as np, pandas as pd, sys
from scipy.stats import ks_2samp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chhal.contract import LABEL_COLUMN
from chhal.data import load_base_data
from chhal.fidelity import CONTROLLED_FEATURES
te = load_base_data(source="ieee").test
legit = te[te[LABEL_COLUMN]==0]; fraud = te[te[LABEL_COLUMN]==1]
print("CONTROLLED_FEATURES:", CONTROLLED_FEATURES)
print("n legit", len(legit), "n fraud", len(fraud))
def mks(ref, samp):
    return float(np.mean([ks_2samp(ref[c].to_numpy(), samp[c].to_numpy()).statistic for c in CONTROLLED_FEATURES]))
print("real fraud full n=%d: %.4f" % (len(fraud), mks(legit, fraud)))
rng = np.random.default_rng(0)

# The legit-vs-legit null floor must be TWO INDEPENDENT legit samples. The earlier version
# used ref=FULL legit vs samp=legit.sample(n), i.e. the sample was a SUBSET of the
# reference. Shared rows constrain the two ECDFs toward each other and bias the KS DOWNWARD
# by ~1.4x (verified: 0.038 overlapping vs 0.053 disjoint at n=500 on a standard normal;
# ~1.44x on a 3-category discrete feature). Draw two DISJOINT legit sub-samples instead.
# 15 reps also under-powered the estimate (~10% run-to-run on the mean); use 100 -> ~1%.
REPS = 100
def disjoint_legit(n):
    """Two independent, non-overlapping legit sub-samples of size n each."""
    idx = rng.permutation(len(legit))
    return legit.iloc[idx[:n]], legit.iloc[idx[n:2*n]]

for n in (500, 1000, 5100):
    v, v0 = [], []
    for _ in range(REPS):
        a, b = disjoint_legit(n)                 # a = legit reference draw, b = legit probe
        v0.append(mks(a, b))                     # H0: legit-vs-legit null floor (n vs n)
        v.append(mks(a, fraud.sample(n=min(n,len(fraud)),
                                     random_state=int(rng.integers(1e6)))))  # H1: same geometry
    print(f"n={n:5d}  real-fraud mean_ks={np.mean(v):.4f} (sd {np.std(v):.4f})   "
          f"NULL floor legit-vs-legit={np.mean(v0):.4f} (sd {np.std(v0):.4f})")

# per-feature null floor at n=m=500. Two DISJOINT 500-row legit samples. This is the
# measurement that anchors KS_NULL_FLOOR / KS_NULL_FLOOR_N=500 in chhal/fidelity.py, so it
# must be done at n=m=500 with independent samples (ks_null_floor() rescales to other n,m).
print("\nper-feature NULL floor at n=m=500 (two INDEPENDENT legit samples, %d reps):" % REPS)
for c in CONTROLLED_FEATURES:
    col = legit[c].to_numpy()
    vals = []
    for _ in range(REPS):
        idx = rng.permutation(len(col))
        vals.append(ks_2samp(col[idx[:500]], col[idx[500:1000]]).statistic)
    print(f"  {c:26s} nunique={legit[c].nunique():7d}  null_ks={np.mean(vals):.4f} (sd {np.std(vals):.4f})")
