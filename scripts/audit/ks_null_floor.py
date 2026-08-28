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
for n in (500, 1000, 5100):
    # H1: real fraud subsampled
    v = [mks(legit, fraud.sample(n=min(n,len(fraud)), random_state=int(rng.integers(1e6)))) for _ in range(15)]
    # H0: legit vs legit (null floor at that n)
    v0 = [mks(legit, legit.sample(n=n, random_state=int(rng.integers(1e6)))) for _ in range(15)]
    print(f"n={n:5d}  real-fraud mean_ks={np.mean(v):.4f} (sd {np.std(v):.4f})   NULL floor legit-vs-legit={np.mean(v0):.4f} (sd {np.std(v0):.4f})")
# per-feature null floor at n=500
print("\nper-feature NULL floor at n=500 (legit vs legit):")
for c in CONTROLLED_FEATURES:
    vals=[ks_2samp(legit[c].to_numpy(), legit[c].sample(500,random_state=i).to_numpy()).statistic for i in range(10)]
    nun = legit[c].nunique()
    print(f"  {c:26s} nunique={nun:7d}  null_ks={np.mean(vals):.4f}")
