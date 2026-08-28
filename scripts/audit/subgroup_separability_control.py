import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from chhal.contract import FEATURE_COLUMNS, LABEL_COLUMN
from chhal.data import load_base_data
base = load_base_data(source="ieee")
fraud = base.test[base.test[LABEL_COLUMN]==1]
legit = base.test[base.test[LABEL_COLUMN]==0]
def auc(A,B,n=200):
    X = pd.concat([A[FEATURE_COLUMNS],B[FEATURE_COLUMNS]],ignore_index=True)
    y = np.r_[np.ones(len(A)),np.zeros(len(B))]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.3,random_state=0,stratify=y)
    m = lgb.LGBMClassifier(n_estimators=n,verbose=-1).fit(Xtr,ytr)
    return roc_auc_score(yte,m.predict_proba(Xte)[:,1])
print("CONTROL: is a NARROW REAL subgroup trivially separable from the rest of real fraud?")
for c in sorted(fraud.channel_code.unique()):
    A = fraud[fraud.channel_code==c]; B = fraud[fraud.channel_code!=c]
    if len(A)<200 or len(B)<200: continue
    print(f"  real fraud channel_code=={c} (n={len(A)}) vs rest (n={len(B)}): AUC={auc(A,B):.4f}")
# random split of real fraud = pure null
A = fraud.sample(frac=.5, random_state=1); B = fraud.drop(A.index)
print(f"  NULL: random half of real fraud vs other half: AUC={auc(A,B):.4f}")
# narrow real-fraud subgroup defined jointly (like a vector): high velocity + new payee
A = fraud[(fraud.velocity_24h>=fraud.velocity_24h.quantile(.8)) & (fraud.is_new_beneficiary==1)]
B = fraud.drop(A.index)
print(f"  narrow joint real subgroup (n={len(A)}) vs rest real fraud (n={len(B)}): AUC={auc(A,B):.4f}")
print(f"  same narrow real subgroup vs LEGIT: AUC={auc(A, legit.sample(5000,random_state=0)):.4f}")
