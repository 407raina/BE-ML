"""
ensemble.py - Network Intrusion Detection (NSL-KDD)
Combine SVM + Naive Bayes into a single voting ensemble, and check
whether combining them actually beats the best individual approach
from imbalance_handling.py.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import time

# ---------------------------------------------------------------
# 1. Load preprocessing.py's processed data
# ---------------------------------------------------------------
train_df = pd.read_csv("processed_train.csv")
test_df = pd.read_csv("processed_test.csv")

target_col = "attack_category"
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# ---------------------------------------------------------------
# 2. Prepare training data for each base model, using whichever
#    imbalance-handling approach worked best for it in imbalance_handling.py:
#    - Naive Bayes: trained on SMOTE-balanced data (it needs the
#      extra examples since it doesn't support class_weight)
#    - SVM: trained on original data with class_weight='balanced'
#      (imbalance_handling.py showed this gave the best U2R precision/F1 for SVM)
# ---------------------------------------------------------------
print("Preparing training data for each base model...")

smote_targets = {"R2L": 5000, "U2R": 2000}
smote = SMOTE(sampling_strategy=smote_targets, random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

SVM_SAMPLE_SIZE = 20000
sample_idx = (
    train_df.groupby(target_col, group_keys=False)
    .apply(lambda g: g.sample(
        frac=min(1.0, SVM_SAMPLE_SIZE / len(train_df)), random_state=42
    ))
    .index
)
X_train_svm = X_train.loc[sample_idx]
y_train_svm = y_train.loc[sample_idx]

# ---------------------------------------------------------------
# 3. Train each base model SEPARATELY first, so we can look at
#    them individually before combining (sanity check + comparison)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Training individual base models...")

nb_model = GaussianNB()
nb_model.fit(X_train_smote, y_train_smote)
nb_preds = nb_model.predict(X_test)

svm_model = SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")
svm_model.fit(X_train_svm, y_train_svm)
svm_preds = svm_model.predict(X_test)

print("Individual models trained.")

# ---------------------------------------------------------------
# 4. Build the VOTING ENSEMBLE
#
# "Soft" voting: instead of each model just shouting one class name,
# each model outputs a PROBABILITY for every class (e.g. SVM might
# say "70% DoS, 20% Probe, 10% other"). The ensemble averages these
# probabilities across both models and picks whichever class has
# the highest averaged probability. This is more informative than
# "hard" voting (which only counts final guesses, ignoring confidence).
#
# IMPORTANT PRACTICAL ISSUE: VotingClassifier's fit() would normally
# retrain both models itself on ONE shared dataset - but we specifically
# want each base model trained on ITS OWN best dataset from imbalance_handling.py
# (NB on SMOTE data, SVM on class-weighted original data). So we use
# VotingClassifier with prefit-style manual combination instead of
# calling ensemble.fit() directly.
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Combining into a soft-voting ensemble (manual, using each model's")
print("own best-prepared training data from imbalance_handling.py)...")

nb_proba = nb_model.predict_proba(X_test)
svm_proba = svm_model.predict_proba(X_test)

# average the two probability tables, then pick the highest for each row
class_labels = nb_model.classes_  # both models see the same 5 classes
avg_proba = (nb_proba + svm_proba) / 2
ensemble_preds = class_labels[np.argmax(avg_proba, axis=1)]

print("\n--- Voting Ensemble: Classification Report ---")
print(classification_report(y_test, ensemble_preds, zero_division=0))

labels_order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
print("\n--- Voting Ensemble Confusion Matrix ---")
print("Rows = actual, Columns = predicted, order:", labels_order)
print(confusion_matrix(y_test, ensemble_preds, labels=labels_order))

# ---------------------------------------------------------------
# 5. Full comparison: baseline_models.py vs imbalance_handling.py vs this ensemble
# ---------------------------------------------------------------
def get_metrics(y_true, y_pred, label):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    m = report.get(label, {"precision": 0, "recall": 0, "f1-score": 0})
    overall_acc = report["accuracy"]
    return m, overall_acc

print("\n" + "=" * 60)
print("FULL SUMMARY: all approaches, baseline_models.py through ensemble.py")
print("=" * 60)

phase2_svm_baseline = {"R2L": {"precision": 0.99, "recall": 0.09, "f1-score": 0.16},
                        "U2R": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00}}
phase2_acc = 0.75

# Recompute imbalance_handling.py-equivalent results fresh in this script for consistency
r2l_nb, nb_acc = get_metrics(y_test, nb_preds, "R2L")
u2r_nb, _ = get_metrics(y_test, nb_preds, "U2R")
r2l_svm, svm_acc = get_metrics(y_test, svm_preds, "R2L")
u2r_svm, _ = get_metrics(y_test, svm_preds, "U2R")
r2l_ens, ens_acc = get_metrics(y_test, ensemble_preds, "R2L")
u2r_ens, _ = get_metrics(y_test, ensemble_preds, "U2R")

final_results = {
    "baseline_models.py: SVM baseline (no fix)": (phase2_svm_baseline["R2L"], phase2_svm_baseline["U2R"], phase2_acc),
    "imbalance_handling.py: Naive Bayes + SMOTE": (r2l_nb, u2r_nb, nb_acc),
    "imbalance_handling.py: SVM + class_weight='balanced'": (r2l_svm, u2r_svm, svm_acc),
    "ensemble.py: Voting Ensemble (SVM + NB)": (r2l_ens, u2r_ens, ens_acc),
}

print(f"\n{'Approach':<42} {'Overall Acc':<12} {'R2L P/R/F1':<18} {'U2R P/R/F1':<18}")
print("-" * 92)
for name, (r2l, u2r, acc) in final_results.items():
    r2l_str = f"{r2l['precision']:.2f}/{r2l['recall']:.2f}/{r2l['f1-score']:.2f}"
    u2r_str = f"{u2r['precision']:.2f}/{u2r['recall']:.2f}/{u2r['f1-score']:.2f}"
    print(f"{name:<42} {acc:<12.2f} {r2l_str:<18} {u2r_str:<18}")

