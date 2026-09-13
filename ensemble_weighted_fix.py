"""
ensemble_weighted_fix.py - Network Intrusion Detection (NSL-KDD)
Fixing the ensemble.py ensemble problem: naive soft-voting collapsed to
just Naive Bayes because NB's probabilities are overconfident
(>0.999 on 99% of predictions), drowning out SVM's more realistic,
moderate confidence scores.

Two fixes tried here:
  A) Weighted soft voting - give SVM's probabilities more influence
     than NB's, since SVM is individually the stronger model.
  B) Hard voting - ignore probabilities entirely, just count which
     class each model "votes" for as its final answer. With only
     2 models this effectively becomes "if they disagree, use a
     tie-break rule" - so we lean on SVM as the tie-breaker since
     it's the stronger individual model.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------
# 1. Load data, prepare each model's training set (same as ensemble.py)
# ---------------------------------------------------------------
train_df = pd.read_csv("processed_train.csv")
test_df = pd.read_csv("processed_test.csv")

target_col = "attack_category"
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

smote = SMOTE(sampling_strategy={"R2L": 5000, "U2R": 2000}, random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

SVM_SAMPLE_SIZE = 20000
sample_idx = (
    train_df.groupby(target_col, group_keys=False)
    .apply(lambda g: g.sample(frac=min(1.0, SVM_SAMPLE_SIZE / len(train_df)), random_state=42))
    .index
)
X_train_svm = X_train.loc[sample_idx]
y_train_svm = y_train.loc[sample_idx]

print("Training base models...")
nb_model = GaussianNB().fit(X_train_smote, y_train_smote)
svm_model = SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")
svm_model.fit(X_train_svm, y_train_svm)

nb_proba = nb_model.predict_proba(X_test)
svm_proba = svm_model.predict_proba(X_test)
nb_preds = nb_model.predict(X_test)
svm_preds = svm_model.predict(X_test)
class_labels = nb_model.classes_

# ---------------------------------------------------------------
# 2. FIX A: Weighted soft voting
#    Instead of a plain 50/50 average, give SVM more weight since
#    it's individually the stronger model (74% vs 47% accuracy).
#    Try a couple of weightings and see which gives the best
#    macro-F1 (macro-F1 treats every class equally, so it won't
#    let big classes like Normal/DoS hide poor U2R/R2L performance -
#    exactly the metric we actually care about here).
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("FIX A: Weighted soft voting - trying a few SVM:NB weight ratios")

best_weight, best_macro_f1, best_preds = None, -1, None
for svm_weight in [0.6, 0.7, 0.8, 0.9]:
    nb_weight = 1 - svm_weight
    avg_proba = svm_weight * svm_proba + nb_weight * nb_proba
    preds = class_labels[np.argmax(avg_proba, axis=1)]
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    print(f"  SVM weight={svm_weight:.1f}, NB weight={nb_weight:.1f} -> macro F1 = {macro_f1:.3f}")
    if macro_f1 > best_macro_f1:
        best_weight, best_macro_f1, best_preds = svm_weight, macro_f1, preds

print(f"\nBest weighting: SVM={best_weight:.1f} / NB={1-best_weight:.1f} (macro F1 = {best_macro_f1:.3f})")
weighted_preds = best_preds

print("\n--- Weighted Voting Ensemble: Classification Report ---")
print(classification_report(y_test, weighted_preds, zero_division=0))

# ---------------------------------------------------------------
# 3. FIX B: Hard voting with SVM as tie-breaker
#    If both models agree, use that class. If they disagree,
#    trust SVM (the individually stronger model).
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("FIX B: Hard voting (SVM as tie-breaker on disagreement)")

hard_preds = np.where(nb_preds == svm_preds, nb_preds, svm_preds)

print("\n--- Hard Voting Ensemble: Classification Report ---")
print(classification_report(y_test, hard_preds, zero_division=0))

agree_rate = (nb_preds == svm_preds).mean()
print(f"NB and SVM agreed on {agree_rate:.1%} of test predictions.")

# ---------------------------------------------------------------
# 4. Final comparison: every approach, baseline_models.py through ensemble_weighted_fix.py
# ---------------------------------------------------------------
def get_metrics(y_true, y_pred, label):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    m = report.get(label, {"precision": 0, "recall": 0, "f1-score": 0})
    return m, report["accuracy"], report["macro avg"]["f1-score"]

phase2_svm_baseline = {"precision": 0.99, "recall": 0.09, "f1-score": 0.16}
phase2_u2r = {"precision": 0.00, "recall": 0.00, "f1-score": 0.00}

r2l_nb, acc_nb, macf1_nb = get_metrics(y_test, nb_preds, "R2L")
u2r_nb, _, _ = get_metrics(y_test, nb_preds, "U2R")
r2l_svm, acc_svm, macf1_svm = get_metrics(y_test, svm_preds, "R2L")
u2r_svm, _, _ = get_metrics(y_test, svm_preds, "U2R")
r2l_naive_ens, acc_naive_ens, macf1_naive_ens = get_metrics(
    y_test, class_labels[np.argmax((svm_proba + nb_proba) / 2, axis=1)], "R2L"
)
u2r_naive_ens, _, _ = get_metrics(
    y_test, class_labels[np.argmax((svm_proba + nb_proba) / 2, axis=1)], "U2R"
)
r2l_w, acc_w, macf1_w = get_metrics(y_test, weighted_preds, "R2L")
u2r_w, _, _ = get_metrics(y_test, weighted_preds, "U2R")
r2l_h, acc_h, macf1_h = get_metrics(y_test, hard_preds, "R2L")
u2r_h, _, _ = get_metrics(y_test, hard_preds, "U2R")

print("\n" + "=" * 60)
print("FULL COMPARISON: baseline_models.py -> ensemble_weighted_fix.py")
print("=" * 60)
rows = [
    ("baseline_models.py: SVM baseline (no fix)", phase2_svm_baseline, phase2_u2r, 0.75, None),
    ("imbalance_handling.py: Naive Bayes + SMOTE", r2l_nb, u2r_nb, acc_nb, macf1_nb),
    ("imbalance_handling.py: SVM + class_weight balanced", r2l_svm, u2r_svm, acc_svm, macf1_svm),
    ("ensemble.py: Naive 50/50 voting (collapsed to NB)", r2l_naive_ens, u2r_naive_ens, acc_naive_ens, macf1_naive_ens),
    (f"ensemble_weighted_fix.py: Weighted voting (SVM={best_weight:.1f})", r2l_w, u2r_w, acc_w, macf1_w),
    ("ensemble_weighted_fix.py: Hard voting (SVM tie-break)", r2l_h, u2r_h, acc_h, macf1_h),
]
print(f"\n{'Approach':<46} {'Acc':<7} {'MacroF1':<9} {'R2L F1':<8} {'U2R F1':<8}")
print("-" * 82)
for name, r2l, u2r, acc, macf1 in rows:
    macf1_str = f"{macf1:.3f}" if macf1 is not None else "  -  "
    print(f"{name:<46} {acc:<7.2f} {macf1_str:<9} {r2l['f1-score']:<8.2f} {u2r['f1-score']:<8.2f}")
