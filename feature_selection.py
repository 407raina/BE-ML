"""
feature_selection.py - Network Intrusion Detection (KDD99) Feature Selection using the Firefly Algorithm 

WHAT IS THE FIREFLY ALGORITHM (plain terms, also explained in the
accompanying writeup): it's a nature-inspired search algorithm. Each
"firefly" represents one candidate subset of features. Brighter fireflies
(= better feature subsets, measured by classifier accuracy) attract
dimmer ones, so the population gradually converges toward good feature
subsets, with some randomness so it doesn't get stuck on a mediocre subset.

PRACTICAL NOTE ON SPEED: evaluating a firefly's "brightness" means training
a classifier on that feature subset. To keep the search fast enough to
actually run, we use Gaussian Naive Bayes (very fast) as the SEARCH-TIME
fitness classifier, and simple label-encoding (not full one-hot) for the
3 categorical columns during the search only. The FINAL model training in
ensemble_replication.py will properly one-hot encode whichever categorical columns get
selected - this file only decides WHICH of the 41 original features to
keep.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Load data_preprocessing.py's raw (unencoded, unscaled) train data
# ---------------------------------------------------------------
train_df = pd.read_csv("raw_train.csv")
target_col = "attack_category"

feature_cols = [c for c in train_df.columns if c != target_col]
print(f"Starting with {len(feature_cols)} original features.")

# Quick label-encoding of the 3 categorical columns, JUST for the search
# (fast + simple; the real pipeline uses proper one-hot encoding later)
search_df = train_df.copy()
categorical_cols = ["protocol_type", "service", "flag"]
for col in categorical_cols:
    search_df[col] = LabelEncoder().fit_transform(search_df[col])

X = search_df[feature_cols].values.astype(float)
y = search_df[target_col].values

# Split off a small validation set purely for scoring fireflies during
# the search (separate from the real ensemble_replication.py test set - no leakage)
X_fit, X_val, y_fit, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ---------------------------------------------------------------
# 2. Fitness function
#    Score = macro-F1 (treats all 5 classes equally - we care about
#    U2R/R2L, not just the big classes) MINUS a small penalty per
#    feature used, so the algorithm is nudged toward smaller,
#    similarly-performing subsets (mirrors the paper's 30/41 result).
# ---------------------------------------------------------------
N_FEATURES = len(feature_cols)
FEATURE_PENALTY = 0.003  # small cost per selected feature

def fitness(binary_mask):
    selected = np.where(binary_mask == 1)[0]
    if len(selected) == 0:
        return 0.0  # a useless firefly with no features selected
    clf = GaussianNB()
    clf.fit(X_fit[:, selected], y_fit)
    preds = clf.predict(X_val[:, selected])
    macro_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    penalty = FEATURE_PENALTY * len(selected)
    return macro_f1 - penalty

# ---------------------------------------------------------------
# 3. Binary Firefly Algorithm
#
# Each firefly = a real-valued vector in [0,1]^41 (its "position").
# We convert position -> binary feature mask via a 0.5 threshold.
# Brightness = fitness(binary mask).
# A dimmer firefly moves toward a brighter one; movement strength
# decreases with distance (attractiveness), plus a small random
# nudge so the search keeps exploring instead of collapsing early.
# ---------------------------------------------------------------
POP_SIZE = 12
GENERATIONS = 20
ALPHA = 0.3       # randomness strength
BETA0 = 1.0        # base attractiveness
GAMMA = 1.0        # light absorption coefficient (controls attractiveness decay)

positions = np.random.rand(POP_SIZE, N_FEATURES)

def to_binary(pos):
    return (pos > 0.5).astype(int)

print("\nRunning Firefly Algorithm for feature selection...")
print(f"Population size: {POP_SIZE}, Generations: {GENERATIONS}")
start = time.time()

brightness = np.array([fitness(to_binary(positions[i])) for i in range(POP_SIZE)])
best_history = []

for gen in range(GENERATIONS):
    for i in range(POP_SIZE):
        for j in range(POP_SIZE):
            if brightness[j] > brightness[i]:
                r = np.linalg.norm(positions[i] - positions[j])
                beta = BETA0 * np.exp(-GAMMA * r ** 2)
                rand_term = ALPHA * (np.random.rand(N_FEATURES) - 0.5)
                positions[i] = positions[i] + beta * (positions[j] - positions[i]) + rand_term
                positions[i] = np.clip(positions[i], 0, 1)
                brightness[i] = fitness(to_binary(positions[i]))
    best_history.append(brightness.max())
    if (gen + 1) % 5 == 0 or gen == 0:
        print(f"  Generation {gen+1}/{GENERATIONS} - best fitness so far: {brightness.max():.4f}")

print(f"\nFirefly search completed in {time.time() - start:.1f}s")

# ---------------------------------------------------------------
# 4. Extract the best feature subset found
# ---------------------------------------------------------------
best_idx = np.argmax(brightness)
best_mask = to_binary(positions[best_idx])
selected_features = [feature_cols[i] for i in range(N_FEATURES) if best_mask[i] == 1]

print(f"\nSelected {len(selected_features)} of {N_FEATURES} features "
      f"(best fitness = {brightness[best_idx]:.4f}):")
for f in selected_features:
    print(" -", f)

# ---------------------------------------------------------------
# 5. Compare against using ALL features (sanity check - did
#    selection actually help, or at least not hurt much, while
#    using fewer features?)
# ---------------------------------------------------------------
all_features_fitness = fitness(np.ones(N_FEATURES, dtype=int))
print(f"\nFor comparison, fitness using ALL {N_FEATURES} features: {all_features_fitness + FEATURE_PENALTY * N_FEATURES:.4f}")
print(f"(macro-F1 without the feature-count penalty, for a fair comparison)")
selected_macro_f1 = brightness[best_idx] + FEATURE_PENALTY * len(selected_features)
print(f"Selected {len(selected_features)}-feature subset macro-F1: {selected_macro_f1:.4f}")

# ---------------------------------------------------------------
# 6. Save the selected feature list for ensemble_replication.py
# ---------------------------------------------------------------
with open("selected_features.txt", "w") as f:
    f.write("\n".join(selected_features))
print("\nSaved selected_features.txt for ensemble_replication.py.")
