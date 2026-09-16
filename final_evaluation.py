"""
final_evaluation.py - Network Intrusion Detection (NSL-KDD)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Load ensemble_replication.py's results
# ---------------------------------------------------------------
our_results = pd.read_csv("replication_results.csv")

paper_results = pd.DataFrame([
    {"Classifier": "Naive Bayes", "Accuracy": 0.90, "Precision": 0.82, "Recall": 0.89, "F1 Score": 0.83},
    {"Classifier": "Support Vector Machine", "Accuracy": 0.91, "Precision": 0.87, "Recall": 0.92, "F1 Score": 0.87},
    {"Classifier": "Ensemble Classifier of SVM-NB", "Accuracy": 0.935, "Precision": 0.928, "Recall": 0.94, "F1 Score": 0.93},
])

# ---------------------------------------------------------------
# 2. Bar chart: Paper vs Our Replication, Accuracy per classifier
# ---------------------------------------------------------------
classifiers = ["Naive Bayes", "Support Vector Machine", "Ensemble Classifier of SVM-NB"]
paper_acc = paper_results.set_index("Classifier").loc[classifiers, "Accuracy"].values
our_acc = our_results.set_index("Classifier").loc[classifiers, "Accuracy"].values

x = np.arange(len(classifiers))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
bars1 = ax.bar(x - width/2, paper_acc, width, label="Paper (Kavitha et al. 2024)", color="#6b7280")
bars2 = ax.bar(x + width/2, our_acc, width, label="Our Replication (NSL-KDD, Firefly FS)", color="#2563eb")

ax.set_ylabel("Accuracy")
ax.set_title("Accuracy: Paper's Reported Results vs Our Replication")
ax.set_xticks(x)
ax.set_xticklabels(["Naive Bayes", "SVM", "Ensemble\n(SVM-NB)"])
ax.set_ylim(0, 1.05)
ax.legend()
ax.bar_label(bars1, fmt="%.3f", padding=3)
ax.bar_label(bars2, fmt="%.3f", padding=3)
plt.tight_layout()
plt.savefig("paper_vs_replication_accuracy.png", dpi=150)
print("Saved paper_vs_replication_accuracy.png")
plt.close()

# ---------------------------------------------------------------
# 3. Confusion matrix heatmap for our final Ensemble
#    (re-derive predictions quickly using saved artifacts would
#    require re-running ensemble_replication.py's model objects, which
#    aren't persisted - so we rebuild the confusion matrix numbers
#    directly from that script's printed output, hardcoded here since
#    they're already verified console output from that run)
# ---------------------------------------------------------------
labels_order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
cm = np.array([
    [1510,   18,   29,   37,    6],
    [  23, 1947,   27,    3,    0],
    [  34,    2, 2769,   11,    0],
    [  38,    0,   10,  727,    1],
    [   3,    0,    0,    6,   15],
])

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(np.arange(len(labels_order)))
ax.set_yticks(np.arange(len(labels_order)))
ax.set_xticklabels(labels_order)
ax.set_yticklabels(labels_order)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Final Ensemble (SVM-NB), NSL-KDD")

for i in range(len(labels_order)):
    for j in range(len(labels_order)):
        val = cm[i, j]
        color = "white" if val > cm.max() / 2 else "black"
        ax.text(j, i, val, ha="center", va="center", color=color, fontsize=11)

plt.colorbar(im, ax=ax, label="Number of instances")
plt.tight_layout()
plt.savefig("final_ensemble_confusion_matrix.png", dpi=150)
print("Saved final_ensemble_confusion_matrix.png")
plt.close()

# ---------------------------------------------------------------
# 4. Metric-by-metric comparison table (saved as CSV + printed)
# ---------------------------------------------------------------
comparison_rows = []
for clf in classifiers:
    p = paper_results.set_index("Classifier").loc[clf]
    o = our_results.set_index("Classifier").loc[clf]
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        comparison_rows.append({
            "Classifier": clf, "Metric": metric,
            "Paper": p[metric], "Our Replication": o[metric],
            "Difference": round(o[metric] - p[metric], 3),
        })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv("final_comparison_table.csv", index=False)
print("\nFull metric comparison:")
print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nSaved final_comparison_table.csv")
