"""
data_preprocessing.py
Approach 2: Paper-replication methodology (Kavitha et al., IJISAE 2024),
applied to the NSL-KDD dataset (instead of the paper's original KDD99),
so this approach uses the SAME dataset as Approach 1 for a fair,
apples-to-apples comparison between "our own exploration" (Approach 1)
and "the paper's method" (Approach 2).

Steps: load NSL-KDD, map attack labels into 5 categories, apply a
paper-style sampling strategy (keep all rare-class instances, subsample
the huge DoS/Normal classes), then do an 80/20 split - matching the
paper's train/test methodology.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# 1. Load NSL-KDD (both train and test files, since we're doing our
#    OWN 80/20 split to match the paper's methodology, rather than
#    using NSL-KDD's own predefined split as Approach 1 did)
# ---------------------------------------------------------------
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

train_raw = pd.read_csv("../data/KDDTrain+.txt", names=columns)
test_raw = pd.read_csv("../data/KDDTest+.txt", names=columns)
df = pd.concat([train_raw, test_raw], ignore_index=True).drop(columns=["difficulty"])
print("Combined NSL-KDD shape (train+test merged, we'll re-split ourselves):", df.shape)

# ---------------------------------------------------------------
# 2. Map attack labels into 5 categories (Normal, DoS, Probe, R2L, U2R)
# ---------------------------------------------------------------
attack_mapping = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS", "processtable": "DoS",
    "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}
df["attack_category"] = df["label"].map(attack_mapping).fillna("Unknown")
df = df.drop(columns=["label"])
df = df[df["attack_category"] != "Unknown"]  # drop the handful of unmapped labels

print("\nFull mapped category distribution:")
print(df["attack_category"].value_counts())

# ---------------------------------------------------------------
# 3. Paper-style sampling: keep ALL rare-class instances (Probe, R2L,
#    U2R), subsample the two huge classes (DoS, Normal) down to a
#    manageable, more balanced size - same philosophy as the paper's
#    deliberately near-balanced 10,000-instance subsample.
# ---------------------------------------------------------------
DOS_SAMPLE = 10000
NORMAL_SAMPLE = 8000

rare = df[df["attack_category"].isin(["Probe", "R2L", "U2R"])]
dos_sample = df[df["attack_category"] == "DoS"].sample(n=DOS_SAMPLE, random_state=42)
normal_sample = df[df["attack_category"] == "Normal"].sample(n=NORMAL_SAMPLE, random_state=42)

df_sample = pd.concat([rare, dos_sample, normal_sample]).sample(frac=1, random_state=42)
print(f"\nFinal sample size: {len(df_sample)} (all rare-class instances kept)")
print(df_sample["attack_category"].value_counts())

# ---------------------------------------------------------------
# 4. 80/20 train/test split (matching the paper exactly)
# ---------------------------------------------------------------
train_df, test_df = train_test_split(
    df_sample, test_size=0.2, stratify=df_sample["attack_category"], random_state=42
)
print(f"\nTrain shape: {train_df.shape}, Test shape: {test_df.shape}")

# ---------------------------------------------------------------
# 5. Save RAW (unencoded, unscaled) splits - encoding/scaling happens
#    AFTER feature selection in feature_selection.py, since selection
#    should operate on the original 41 features, not on expanded
#    one-hot columns.
# ---------------------------------------------------------------
train_df.to_csv("raw_train.csv", index=False)
test_df.to_csv("raw_test.csv", index=False)
print("\nSaved raw_train.csv / raw_test.csv - ready for feature_selection.py.")
