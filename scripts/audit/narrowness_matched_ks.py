import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scipy.stats import ks_2samp
from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN
from chhal.data import load_base_data
from chhal.fidelity import CONTROLLED_FEATURES
base = load_base_data(source="ieee")
fraud = base.test[base.test[LABEL_COLUMN]==1]; legit = base.test[base.test[LABEL_COLUMN]==0]
def mks(s):
    return float(np.mean([ks_2samp(legit[c].to_numpy(), s[c].to_numpy()).statistic for c in CONTROLLED_FEATURES]))
print("DECISIVE CONTROL: mean_ks_controlled vs legit for NARROW REAL fraud subgroups")
print(f"  ALL real test fraud                       n={len(fraud):5d}  ks={mks(fraud):.4f}")
subs = {
 "channel_code==0": fraud[fraud.channel_code==0],
 "channel_code==1": fraud[fraud.channel_code==1],
 "channel_code==2": fraud[fraud.channel_code==2],
 "new payee only": fraud[fraud.is_new_beneficiary==1],
 "cross-border only": fraud[fraud.is_cross_border==1],
 "top-quintile velocity_24h": fraud[fraud.velocity_24h>=fraud.velocity_24h.quantile(.8)],
 "burst: vel24h>=q80 & gap<q20": fraud[(fraud.velocity_24h>=fraud.velocity_24h.quantile(.8))&(fraud.time_since_last_txn_min<=fraud.time_since_last_txn_min.quantile(.2))],
 "high-amount q80+": fraud[fraud.amount>=fraud.amount.quantile(.8)],
 "night hours 0-5": fraud[fraud.hour<6],
}
for k,v in subs.items():
    if len(v)<50: print(f"  {k:42s} n={len(v):5d}  (too small)"); continue
    print(f"  {k:42s} n={len(v):5d}  ks={mks(v):.4f}")
print("\nfor reference, chhal vectors (optimized): threshold_hugging 0.2501  autopay 0.2507  mule 0.3639  upi 0.4228  bustout 0.4846  card_testing 0.5299")
