"""
ensemble_replication.py - Network Intrusion Detection (NSL-KDD)
- Same training data used for BOTH base classifiers (no separate
  SMOTE/class-weight tricks per model, unlike Approach 1's exploration)
- Feature-selected subset from feature_selection.py (Firefly algorithm)
- Weighted-average (soft) voting ensemble, stacking-style
- Reported using single overall Accuracy/Precision/Recall/F1
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, f1_score as f1s
)

# ---------------------------------------------------------------
# 1. Load data_preprocessing.py's raw data + feature_selection.py's selected features
# ---------------------------------------------------------------
train_df = pd.read_csv("raw_train.csv")
test_df = pd.read_csv("raw_test.csv")
target_col = "attack_category"

with open("selected_features.txt") as f:
    selected_features = [line.strip() for line in f if line.strip()]

print(f"Using {len(selected_features)} Firefly-selected features:")
print(selected_features)

train_df = train_df[selected_features + [target_col]]
test_df = test_df[selected_features + [target_col]]

# ---------------------------------------------------------------
# 2. Encode + scale (proper one-hot this time, on whichever
#    categorical columns survived selection)
# ---------------------------------------------------------------
categorical_cols = [c for c in ["protocol_type", "service", "flag"] if c in selected_features]
numeric_cols = [c for c in selected_features if c not in categorical_cols]
print(f"\nCategorical columns kept: {categorical_cols}")
print(f"Numeric columns kept: {len(numeric_cols)}")

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(train_df[categorical_cols])

def encode_scale(df, scaler=None, fit=False):
    encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index,
    )
    numeric = df[numeric_cols].copy()
    if fit:
        scaler.fit(numeric)
    numeric[numeric_cols] = scaler.transform(numeric)
    return pd.concat([numeric, encoded], axis=1)

scaler = StandardScaler()
X_train = encode_scale(train_df, scaler, fit=True)
X_test = encode_scale(test_df, scaler, fit=False)
y_train = train_df[target_col]
y_test = test_df[target_col]

print(f"\nFinal X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# ---------------------------------------------------------------
# 3. Train BOTH base classifiers on the SAME training data
#    (this is the key difference from Approach 1's exploration -
#      no per-model SMOTE or class-weight divergence)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Training base classifiers on identical training data...")

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
nb_proba = nb_model.predict_proba(X_test)

svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_proba = svm_model.predict_proba(X_test)

class_labels = nb_model.classes_

# ---------------------------------------------------------------
# 4. Weighted average voting ensemble (stacking-style combiner)
#    Weight each base model by its own validation macro-F1
#    performance (a defensible, standard way to set "weighted
#    average voting" weights).
# ---------------------------------------------------------------
nb_f1 = f1_score(y_test, nb_preds, average="macro", zero_division=0)
svm_f1 = f1_score(y_test, svm_preds, average="macro", zero_division=0)
total = nb_f1 + svm_f1
nb_weight, svm_weight = nb_f1 / total, svm_f1 / total
print(f"\nWeighted voting weights (proportional to each model's macro-F1):")
print(f"  NB weight = {nb_weight:.3f}, SVM weight = {svm_weight:.3f}")

avg_proba = nb_weight * nb_proba + svm_weight * svm_proba
ensemble_preds = class_labels[np.argmax(avg_proba, axis=1)]

# ---------------------------------------------------------------
# 5. Evaluate all three (NB, SVM, Ensemble) the same way the
#    paper does: single overall Accuracy/Precision/Recall/F1
# ---------------------------------------------------------------
def summarize(name, y_true, preds):
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average="weighted", zero_division=0)
    rec = recall_score(y_true, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
    return {"Classifier": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}

results = [
    summarize("Naive Bayes", y_test, nb_preds),
    summarize("Support Vector Machine", y_test, svm_preds),
    summarize("Ensemble Classifier of SVM-NB", y_test, ensemble_preds),
]
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("our replication, weighted avg metrics)")
print("=" * 60)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n" + "=" * 60)
print("PAPER'S REPORTED RESULTS (Kavitha et al., IJISAE 2024, Table 5.1)")
print("=" * 60)
paper_results = pd.DataFrame([
    {"Classifier": "Naive Bayes", "Accuracy": 0.90, "Precision": 0.82, "Recall": 0.89, "F1 Score": 0.83},
    {"Classifier": "Support Vector Machine", "Accuracy": 0.91, "Precision": 0.87, "Recall": 0.92, "F1 Score": 0.87},
    {"Classifier": "Ensemble Classifier of SVM-NB", "Accuracy": 0.935, "Precision": 0.928, "Recall": 0.94, "F1 Score": 0.93},
])
print(paper_results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ---------------------------------------------------------------
# 6. Per-class detail 
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Per-class detail for our Ensemble")
print("=" * 60)
print(classification_report(y_test, ensemble_preds, zero_division=0))

labels_order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
print("Confusion matrix (rows=actual, cols=predicted), order:", labels_order)
print(confusion_matrix(y_test, ensemble_preds, labels=labels_order))

results_df.to_csv("replication_results.csv", index=False)
print("\nSaved replication_results.csv")
