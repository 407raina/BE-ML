"""
preprocessing.py - Network Intrusion Detection (NSL-KDD)
Data loading, exploration, attack-category mapping, encoding, scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
# NSL-KDD has 41 features + 1 label column + 1 "difficulty" column, no header.
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

train_df = pd.read_csv("../data/KDDTrain+.txt", names=columns)
test_df = pd.read_csv("../data/KDDTest+.txt", names=columns)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# drop the "difficulty" column - not a feature, just metadata from the dataset creators
train_df = train_df.drop(columns=["difficulty"])
test_df = test_df.drop(columns=["difficulty"])

# ---------------------------------------------------------------
# 2. Basic exploration
# ---------------------------------------------------------------
print("\nRaw label distribution (train):")
print(train_df["label"].value_counts().head(10))

# ---------------------------------------------------------------
# 3. Map ~20+ specific attack labels into 5 categories
#    (Normal, DoS, Probe, R2L, U2R) - this is the standard NSL-KDD mapping
# ---------------------------------------------------------------
attack_mapping = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS", "processtable": "DoS",
    "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}

train_df["attack_category"] = train_df["label"].map(attack_mapping)
test_df["attack_category"] = test_df["label"].map(attack_mapping)

# Any label not in our mapping (rare, dataset-version-specific labels) -> mark Unknown
train_df["attack_category"] = train_df["attack_category"].fillna("Unknown")
test_df["attack_category"] = test_df["attack_category"].fillna("Unknown")

print("\nMapped category distribution (train):")
print(train_df["attack_category"].value_counts())
print("\nMapped category distribution (test):")
print(test_df["attack_category"].value_counts())

# drop original fine-grained label, keep the 5-category label
train_df = train_df.drop(columns=["label"])
test_df = test_df.drop(columns=["label"])

# ---------------------------------------------------------------
# 4. Encode categorical columns (protocol_type, service, flag)
# ---------------------------------------------------------------
categorical_cols = ["protocol_type", "service", "flag"]

# Fit encoder on TRAIN only, then apply to both (avoids data leakage,
# and handles service values in test not seen in train)
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(train_df[categorical_cols])

train_encoded = pd.DataFrame(
    encoder.transform(train_df[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=train_df.index,
)
test_encoded = pd.DataFrame(
    encoder.transform(test_df[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=test_df.index,
)

train_df = pd.concat([train_df.drop(columns=categorical_cols), train_encoded], axis=1)
test_df = pd.concat([test_df.drop(columns=categorical_cols), test_encoded], axis=1)

# ---------------------------------------------------------------
# 5. Scale numeric features
# ---------------------------------------------------------------
target_col = "attack_category"
feature_cols = [c for c in train_df.columns if c != target_col]

scaler = StandardScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# ---------------------------------------------------------------
# 6. Confirm final shape, save processed data for baseline_models.py
# ---------------------------------------------------------------
print("\nFinal processed train shape:", train_df.shape)
print("Final processed test shape:", test_df.shape)

train_df.to_csv("processed_train.csv", index=False)
test_df.to_csv("processed_test.csv", index=False)
print("\nSaved processed_train.csv and processed_test.csv - ready for baseline_models.py (modeling).")
