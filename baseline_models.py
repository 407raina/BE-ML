"""
baseline_models.py - Network Intrusion Detection (NSL-KDD)
Baseline models: SVM and Naive Bayes, evaluated WITHOUT any imbalance handling.
Goal of this phase: establish a baseline and clearly SEE how imbalance hurts
the rare classes (R2L, U2R) before we fix it in imbalance_handling.py.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import time

# ---------------------------------------------------------------
# 1. Load the processed data from preprocessing.py
# ---------------------------------------------------------------
train_df = pd.read_csv("processed_train.csv")
test_df = pd.read_csv("processed_test.csv")

target_col = "attack_category"
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ---------------------------------------------------------------
# 2. Baseline model 1: Naive Bayes
#    Fast to train, good baseline, often struggles when features
#    aren't independent (which network features often aren't) -
#    that's expected and worth noting, not a bug.
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Training Naive Bayes...")
start = time.time()
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
print(f"Naive Bayes trained in {time.time() - start:.1f}s")

print("\n--- Naive Bayes: Classification Report ---")
print(classification_report(y_test, nb_preds, zero_division=0))

# ---------------------------------------------------------------
# 3. Baseline model 2: SVM
#    SVC (RBF kernel) doesn't scale well to 126k rows - training
#    time grows fast with dataset size. For this BASELINE we train
#    on a stratified sample of the training data to keep runtime
#    reasonable; the ensemble in ensemble.py can revisit this tradeoff
#    (e.g. LinearSVC, or a larger sample) if needed.
# ---------------------------------------------------------------
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
print(f"\nSVM training sample size: {len(X_train_svm)} (stratified across classes)")

print("\n" + "=" * 60)
print("Training SVM...")
start = time.time()
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train_svm, y_train_svm)
svm_preds = svm_model.predict(X_test)
print(f"SVM trained in {time.time() - start:.1f}s")

print("\n--- SVM: Classification Report ---")
print(classification_report(y_test, svm_preds, zero_division=0))

# ---------------------------------------------------------------
# 4. Confusion matrices - see exactly WHERE each model gets confused
# ---------------------------------------------------------------
labels_order = ["Normal", "DoS", "Probe", "R2L", "U2R"]

print("\n--- Naive Bayes Confusion Matrix ---")
print("Rows = actual, Columns = predicted, order:", labels_order)
print(confusion_matrix(y_test, nb_preds, labels=labels_order))

print("\n--- SVM Confusion Matrix ---")
print("Rows = actual, Columns = predicted, order:", labels_order)
print(confusion_matrix(y_test, svm_preds, labels=labels_order))

# ---------------------------------------------------------------
# 5. Save predictions for later comparison (imbalance_handling.py / ensemble.py)
# ---------------------------------------------------------------
results_df = pd.DataFrame({
    "actual": y_test,
    "nb_pred": nb_preds,
    "svm_pred": svm_preds,
})
results_df.to_csv("baseline_predictions.csv", index=False)
print("\nSaved baseline_predictions.csv")
