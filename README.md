# Network Intrusion Detection using Ensemble Machine Learning (SVM + Naive Bayes)

Two separate approaches are included, both on the SAME dataset (NSL-KDD), so their results are directly comparable.

## Folder structure

```
network-intrusion-detection/
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── approach-1-imbalance-handling/
│   ├── preprocessing.py
│   ├── baseline_models.py
│   ├── imbalance_handling.py
│   ├── ensemble.py
│   └── ensemble_weighted_fix.py
└── approach-2-paper-replication/
    ├── data_preprocessing.py
    ├── feature_selection.py
    ├── ensemble_replication.py
    └── final_evaluation.py
```

## Approach 1 — Imbalance handling & ensemble exploration

Our own exploration: full NSL-KDD (its original train/test split), no
feature selection, focused on diagnosing and fixing class imbalance
(SMOTE, class-weighting) and diagnosing a real ensemble failure mode.

**Run in this order** (each script reads the CSV the previous one saved):
```bash
cd approach-1-imbalance-handling
python preprocessing.py          # -> processed_train.csv, processed_test.csv
python baseline_models.py        # reads processed_*.csv, no imbalance handling
python imbalance_handling.py     # SMOTE + class_weight='balanced'
python ensemble.py               # naive 50/50 soft voting (shows a real failure)
python ensemble_weighted_fix.py  # weighted voting + hard voting fixes
```

## Approach 2 — Paper-replication methodology

Replicates the methodology of the reference paper — K. Kavitha et al.,
*"A Unified Approach for Network Intrusion Detection using Ensemble
Machine Learning Classifier using Support Vector Machine and Naive
Bayes,"* IJISAE, 12(4), 824-830, 2024 — but applied to **NSL-KDD**
(the paper itself used KDD99) so both approaches in this project sit on
the same dataset.

**Run in this order:**
```bash
cd approach-2-paper-replication
python data_preprocessing.py     # -> raw_train.csv, raw_test.csv (paper-style sampling)
python feature_selection.py      # Firefly algorithm -> selected_features.txt
python ensemble_replication.py   # same-data ensemble, weighted-average voting
python final_evaluation.py       # -> comparison chart + confusion matrix (PNG)
```

### Key result: paper vs. our NSL-KDD replication

| Classifier | Paper (KDD99) | Our Replication (NSL-KDD) |
|---|---|---|
| Naive Bayes | 0.900 | 0.467 |
| SVM | 0.910 | 0.964 |
| Ensemble (SVM-NB) | 0.935 | 0.966 |



## Visualizations in VS Code

`final_evaluation.py` uses `matplotlib.use("Agg")` and `plt.savefig(...)`
— this means it does **not** pop up an interactive window; it silently
saves two PNG files (`paper_vs_replication_accuracy.png`,
`final_ensemble_confusion_matrix.png`) into the folder you ran it from.

**To view them in VS Code:** after running the script, look in the VS
Code file explorer sidebar (left panel) — the new `.png` files will
appear there. Click on either one and VS Code opens a built-in image
preview automatically. No extra setup needed.

**If you'd rather see plots appear inline as you work** (like a
notebook), instead of only as saved files: install the **Jupyter**
extension in VS Code, then add `# %%` above a block of plotting code to
turn it into a runnable "cell" — running that cell opens the plot in
VS Code's Interactive Window directly, without needing `plt.savefig`.
This is optional — the scripts work fine as plain `.py` files either way.

## Setup (once)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

Both approaches read from the shared `data/` folder — make sure
`KDDTrain+.txt` and `KDDTest+.txt` are inside it before running anything.
