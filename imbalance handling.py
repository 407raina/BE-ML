"""
Phase 3 - Network Intrusion Detection (NSL-KDD)
Handling class imbalance with SMOTE and class_weight='balanced',
then comparing against the Phase 2 no-imbalance-handling baseline
(especially for R2L and U2R, which Phase 2 handled very poorly).
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import time

# ---------------------------------------------------------------
# 1. Load Phase 1's processed data
# ---------------------------------------------------------------
train_df = pd.read_csv("processed_train.csv")
test_df = pd.read_csv("processed_test.csv")

target_col = "attack_category"
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

print("Original training class distribution:")
print(y_train.value_counts())

# ---------------------------------------------------------------
# 2. Apply SMOTE to the TRAINING data only
#
# SMOTE creates new SYNTHETIC minority-class rows by interpolating
# between real minority examples (it does NOT just duplicate rows).
# We deliberately do NOT fully equalize every class to match Normal
# (67,343) - that would mean generating ~67,000 synthetic U2R rows
# from only 52 real ones, which is extreme extrapolation and risks
# creating unrealistic, noisy data. Instead we set modest, more
# defensible targets for just the two worst classes.
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Applying SMOTE (targeted, not full equalization)...")

smote_targets = {
    "R2L": 5000,   # up from 995
    "U2R": 2000,   # up from 52 (this is a LOT of extrapolation from 52 points -
                   # worth flagging honestly as a limitation in your report)
}

smote = SMOTE(sampling_strategy=smote_targets, random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nClass distribution AFTER SMOTE:")
print(y_train_smote.value_counts())

# ---------------------------------------------------------------
# 3. Retrain Naive Bayes on SMOTE-balanced data
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Training Naive Bayes on SMOTE-balanced data...")
start = time.time()
nb_smote = GaussianNB()
nb_smote.fit(X_train_smote, y_train_smote)
nb_smote_preds = nb_smote.predict(X_test)
print(f"Trained in {time.time() - start:.1f}s")
print("\n--- Naive Bayes + SMOTE: Classification Report ---")
print(classification_report(y_test, nb_smote_preds, zero_division=0))

# ---------------------------------------------------------------
# 4. Retrain SVM on SMOTE-balanced data
#    Same stratified-sample approach as Phase 2, but now sampling
#    from the SMOTE-balanced set (so rare classes are properly
#    represented in the sample, not just proportionally rare).
# ---------------------------------------------------------------
SVM_SAMPLE_SIZE = 20000
smote_df = X_train_smote.copy()
smote_df[target_col] = y_train_smote.values

sample_idx = (
    smote_df.groupby(target_col, group_keys=False)
    .apply(lambda g: g.sample(
        n=min(len(g), SVM_SAMPLE_SIZE // smote_df[target_col].nunique()),
        random_state=42
    ))
    .index
)
X_train_svm_smote = X_train_smote.loc[sample_idx]
y_train_svm_smote = y_train_smote.loc[sample_idx]
print(f"\nSVM (SMOTE) training sample size: {len(X_train_svm_smote)}")
print(y_train_svm_smote.value_counts())

print("\n" + "=" * 60)
print("Training SVM on SMOTE-balanced data...")
start = time.time()
svm_smote = SVC(kernel="rbf", probability=True, random_state=42)
svm_smote.fit(X_train_svm_smote, y_train_svm_smote)
svm_smote_preds = svm_smote.predict(X_test)
print(f"Trained in {time.time() - start:.1f}s")
print("\n--- SVM + SMOTE: Classification Report ---")
print(classification_report(y_test, svm_smote_preds, zero_division=0))

# ---------------------------------------------------------------
# 5. Alternative: class_weight='balanced' (no SMOTE, no new rows -
#    instead the model is told to "pay more attention" to mistakes
#    on rare classes during training)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Training SVM with class_weight='balanced' (no SMOTE)...")

# use the same stratified sample from Phase 2's original (non-SMOTE) data
# for a fair comparison - re-derive it the same way Phase 2 did
sample_idx_orig = (
    train_df.groupby(target_col, group_keys=False)
    .apply(lambda g: g.sample(
        frac=min(1.0, SVM_SAMPLE_SIZE / len(train_df)), random_state=42
    ))
    .index
)
X_train_svm_orig = X_train.loc[sample_idx_orig]
y_train_svm_orig = y_train.loc[sample_idx_orig]

start = time.time()
svm_weighted = SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")
svm_weighted.fit(X_train_svm_orig, y_train_svm_orig)
svm_weighted_preds = svm_weighted.predict(X_test)
print(f"Trained in {time.time() - start:.1f}s")
print("\n--- SVM + class_weight='balanced': Classification Report ---")
print(classification_report(y_test, svm_weighted_preds, zero_division=0))

# ---------------------------------------------------------------
# 6. Side-by-side comparison: focus on R2L and U2R specifically
# ---------------------------------------------------------------
def get_metrics(y_true, y_pred, label):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report.get(label, {"precision": 0, "recall": 0, "f1-score": 0})

print("\n" + "=" * 60)
print("SUMMARY: R2L and U2R performance across all approaches")
print("=" * 60)

# Recorded directly from Phase 2's actual output (this script runs standalone,
# so we hardcode the earlier result here rather than re-run Phase 2 every time)
phase2_baseline = {"R2L": {"precision": 0.99, "recall": 0.09, "f1-score": 0.16},
                    "U2R": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00}}

approaches = {
    "SVM - Phase 2 baseline (no imbalance handling)": phase2_baseline,
    "SVM + SMOTE": {"R2L": get_metrics(y_test, svm_smote_preds, "R2L"),
                     "U2R": get_metrics(y_test, svm_smote_preds, "U2R")},
    "SVM + class_weight='balanced'": {"R2L": get_metrics(y_test, svm_weighted_preds, "R2L"),
                                        "U2R": get_metrics(y_test, svm_weighted_preds, "U2R")},
    "Naive Bayes + SMOTE": {"R2L": get_metrics(y_test, nb_smote_preds, "R2L"),
                              "U2R": get_metrics(y_test, nb_smote_preds, "U2R")},
}

print(f"\n{'Approach':<48} {'R2L P/R/F1':<20} {'U2R P/R/F1':<20}")
print("-" * 90)
for name, metrics in approaches.items():
    r2l, u2r = metrics["R2L"], metrics["U2R"]
    r2l_str = f"{r2l['precision']:.2f}/{r2l['recall']:.2f}/{r2l['f1-score']:.2f}"
    u2r_str = f"{u2r['precision']:.2f}/{u2r['recall']:.2f}/{u2r['f1-score']:.2f}"
    print(f"{name:<48} {r2l_str:<20} {u2r_str:<20}")
