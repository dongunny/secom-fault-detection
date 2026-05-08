"""
SECOM 반도체 공정 데이터 분석 파이프라인
Semi-Conductor Manufacturing (SECOM) Dataset Analysis

Dataset: 1567 samples x 591 features, binary classification (Pass/Fail)
Result:  LR BER=29.7% (outperforms paper baseline 33.5%)
"""

import os, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve,
                             confusion_matrix, balanced_accuracy_score)
from sklearn.impute import SimpleImputer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'secom')

print("=" * 60)
print("  SECOM 반도체 공정 데이터 분석")
print("=" * 60)

# 1. Data Loading
print("\n[1/5] 데이터 로딩...")
X = pd.read_csv(os.path.join(DATA_DIR, 'secom.data'), sep=' ', header=None, na_values='NaN')
ldf = pd.read_csv(os.path.join(DATA_DIR, 'secom_labels.data'),
                  sep=' ', header=None, names=['label','timestamp'])
y_binary = (ldf['label'].values == 1).astype(int)
print(f"  샘플: {X.shape[0]:,}  변수: {X.shape[1]:,}  불량: {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")

# 2. Preprocessing
print("\n[2/5] 전처리...")
miss = X.isnull().mean()
X = X[miss[miss <= 0.5].index]
X = X.loc[:, X.var() > 0]
imputer = SimpleImputer(strategy='median')
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(imputer.fit_transform(X))
print(f"  최종 변수: {X.shape[1]}개")

# 3. Feature Selection (F-test Top 40)
print("\n[3/5] F-test 특징 선택 (Top 40)...")
selector   = SelectKBest(f_classif, k=40)
X_selected = selector.fit_transform(X_scaled, y_binary)
top_idx    = np.argsort(selector.scores_)[::-1][:20]
print(f"  Top 1 변수: F{X.columns[top_idx[0]]} (F={selector.scores_[top_idx[0]]:.2f})")

# 4. Model Training (10-fold CV)
print("\n[4/5] 모델 학습 (10-fold Stratified CV)...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
models = {
    'Logistic Regression': LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=8, random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    print(f"  [{name}]")
    y_prob = cross_val_predict(model, X_selected, y_binary, cv=cv, method='predict_proba')[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    ber  = 1 - balanced_accuracy_score(y_binary, y_pred)
    auc  = roc_auc_score(y_binary, y_prob)
    cm   = confusion_matrix(y_binary, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_binary, y_prob)
    results[name] = {
        'ber': float(ber*100), 'auc': float(auc),
        'tpr': float(tp/(tp+fn)*100), 'tnr': float(tn/(tn+fp)*100),
        'cm': cm.tolist(),
        'fpr': fpr[::max(1,len(fpr)//100)].tolist(),
        'tpr_curve': tpr[::max(1,len(tpr)//100)].tolist(),
    }
    print(f"    BER={ber*100:.1f}%  AUC={auc:.3f}  TPR={tp/(tp+fn)*100:.1f}%")

# 5. Save Results
print("\n[5/5] 결과 저장...")
rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=8, random_state=42).fit(X_selected, y_binary)
top20_rf = np.argsort(rf_model.feature_importances_)[::-1][:20]
output = {
    'preprocessing': {'total_samples': int(X.shape[0]), 'final_features': 40,
                      'fail_count': int(y_binary.sum()), 'pass_count': int((y_binary==0).sum())},
    'models': results,
    'rf_top20_names':  [f'F{top_idx[i]}' for i in top20_rf],
    'rf_top20_scores': rf_model.feature_importances_[top20_rf].tolist(),
}
with open(os.path.join(RESULTS_DIR, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("  완료!")
for name, res in results.items():
    print(f"  [{name}] BER={res['ber']:.1f}%  AUC={res['auc']:.3f}")
print("=" * 60)
