import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN
from chhal.data import load_base_data
from chhal.detector import Detector
from chhal.evaluation import threshold_for_fpr
from chhal.redteam import ALL_VECTORS
from chhal.redteam.base import BaseProfile
from chhal.redteam.hosts import HostPool
base = load_base_data(source="ieee"); rng = np.random.default_rng(7)
prof = BaseProfile(base.legit_quantiles, base.legit_categoricals)
th = HostPool(base.test, exclude_accounts=base.train['_account'])
det = Detector(seed=7).fit(base.train, LABEL_COLUMN)
legit = base.test[base.test[LABEL_COLUMN]==0]; fraud = base.test[base.test[LABEL_COLUMN]==1]
lp = det.score(legit[FEATURE_COLUMNS]); thr = threshold_for_fpr(lp,0.001)
fp = det.score(fraud[FEATURE_COLUMNS])
atk = pd.concat([V().calibrate(prof,th).batch(500,0,rng).transactions for V in ALL_VECTORS], ignore_index=True)
ap = det.score(atk[FEATURE_COLUMNS])
q = lambda s: float((lp < s.mean()).mean())
print("score percentiles vs legit distribution:")
for nm, s in [("legit",lp),("REAL FRAUD",fp),("synthetic attacks (raw)",ap)]:
    print(f"  {nm:24s} median={np.median(s):.5f}  mean={s.mean():.5f}  pct>=thr={float((s>=thr).mean()):.4f}  median-percentile-in-legit={float((lp<np.median(s)).mean()):.4f}")
# generator fingerprint: separate synthetic attacks from REAL FRAUD
X = pd.concat([atk[FEATURE_COLUMNS], fraud[FEATURE_COLUMNS]], ignore_index=True)
y = np.r_[np.ones(len(atk)), np.zeros(len(fraud))]
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.3,random_state=0,stratify=y)
m = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(Xtr,ytr)
print("\nGENERATOR FINGERPRINT  AUC(synthetic attack vs REAL fraud) = %.4f" % roc_auc_score(yte, m.predict_proba(Xte)[:,1]))
Xl = pd.concat([atk[FEATURE_COLUMNS], legit.sample(3000,random_state=0)[FEATURE_COLUMNS]], ignore_index=True)
yl = np.r_[np.ones(len(atk)), np.zeros(3000)]
Xtr,Xte,ytr,yte = train_test_split(Xl,yl,test_size=.3,random_state=0,stratify=yl)
m2 = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(Xtr,ytr)
print("GENERATOR FINGERPRINT  AUC(synthetic attack vs LEGIT)      = %.4f" % roc_auc_score(yte, m2.predict_proba(Xte)[:,1]))
Xf = pd.concat([fraud[FEATURE_COLUMNS], legit.sample(5100,random_state=0)[FEATURE_COLUMNS]], ignore_index=True)
yf = np.r_[np.ones(len(fraud)), np.zeros(5100)]
Xtr,Xte,ytr,yte = train_test_split(Xf,yf,test_size=.3,random_state=0,stratify=yf)
m3 = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(Xtr,ytr)
print("REFERENCE              AUC(REAL fraud vs LEGIT)            = %.4f" % roc_auc_score(yte, m3.predict_proba(Xte)[:,1]))
