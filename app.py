"""
app.py - Network Intrusion Detection Project Dashboard (Streamlit)
Run with:
    pip install streamlit pandas numpy matplotlib
    streamlit run app.py

All numbers below are hardcoded from ACTUAL executed runs of the
project's Python scripts (preprocessing.py, baseline_models.py,
imbalance_handling.py, ensemble.py, ensemble_weighted_fix.py,
data_preprocessing.py, feature_selection.py, ensemble_replication.py) -
not placeholders.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.size"] = 9

st.set_page_config(
    page_title="Network Intrusion Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------
# Shared color palette (kept consistent across all charts)
# ---------------------------------------------------------------
ACCENT = "#0E8C7C"    # teal - "normal" / our results
ALERT = "#C25A2A"     # amber - "attack" / rare classes
NEUTRAL = "#6B7280"   # grey - paper's reference numbers

st.title("🛡️ Network Intrusion Detection")
st.caption("SVM + Naive Bayes Ensemble Classifier · NSL-KDD dataset")

st.markdown(
    "Classifying network connections as **Normal** or one of four attack "
    "types (**DoS, Probe, R2L, U2R**) using SVM and Naive Bayes, combined "
    "into an ensemble. Two approaches are shown side by side below."
)

# ---------------------------------------------------------------
# Top-level stat row
# ---------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Training connections", "125,973")
c2.metric("Traffic classes", "5")
c3.metric("Features after selection", "41 → 25")
c4.metric("Best ensemble accuracy", "96.6%")

st.divider()

tab_overview, tab_a1, tab_a2, tab_compare, tab_demo = st.tabs(
    ["Overview", "Approach 1 — Imbalance", "Approach 2 — Paper Replica", "Comparison", "Try the Concept"]
)

# =================================================================
# TAB: OVERVIEW
# =================================================================
with tab_overview:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Approach 1 — Imbalance Handling")
        st.markdown(
            "Our own exploration. Full 41-feature space, no feature "
            "selection. Focused on diagnosing why raw accuracy is "
            "misleading on this dataset, and fixing it with SMOTE and "
            "class-weighting."
        )
        st.code(
            "Preprocess → Baseline SVM+NB → Fix imbalance → Ensemble + fix",
            language=None,
        )

    with col2:
        st.subheader("Approach 2 — Paper Replication")
        st.markdown(
            "Reproduces the methodology of Kavitha et al. (IJISAE 2024) "
            "on NSL-KDD instead of their original KDD99: Firefly-algorithm "
            "feature selection, then a same-data SVM+NB ensemble with "
            "weighted voting."
        )
        st.code(
            "Sample + split → Firefly selection → Ensemble (shared data) → Evaluate vs paper",
            language=None,
        )

    st.info(
        "**Why this matters:** Accuracy alone is misleading here — U2R "
        "(privilege escalation) attacks make up under 0.05% of the data. "
        "A model can score 75% accuracy while never once catching a "
        "single U2R attack. Both approaches confront this directly rather "
        "than reporting one flattering number."
    )

# =================================================================
# TAB: APPROACH 1
# =================================================================
with tab_a1:
    st.header("Approach 1 — Imbalance Handling")
    st.caption("Full NSL-KDD, original train/test split, no feature selection.")

    stages = [
        "Baseline SVM", "NB + SMOTE", "SVM + class_weight",
        "Naive ensemble", "Weighted ensemble", "Hard voting",
    ]
    accuracy = [0.75, 0.47, 0.74, 0.47, 0.75, 0.74]
    r2l_f1 = [0.16, 0.27, 0.21, 0.27, 0.02, 0.21]
    u2r_f1 = [0.00, 0.11, 0.22, 0.11, 0.31, 0.22]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(stages))
    width = 0.35
    ax.bar(x - width / 2, u2r_f1, width, label="U2R F1-score", color=ALERT)
    ax.bar(x + width / 2, r2l_f1, width, label="R2L F1-score", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("F1-score")
    ax.set_title("Rare-class F1-score across imbalance-handling stages")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    df1 = pd.DataFrame({
        "Stage": stages,
        "Accuracy": accuracy,
        "R2L F1": r2l_f1,
        "U2R F1": u2r_f1,
        "Note": [
            "Never once catches a U2R attack",
            "Synthetic oversampling of rare classes",
            "Best single-model balance",
            "Collapsed to NB alone",
            "Best U2R, worst R2L",
            "Mathematically equals SVM alone",
        ],
    })
    st.dataframe(df1, use_container_width=True, hide_index=True)

    st.warning(
        "**Finding 1 — accuracy lies.** The baseline SVM hits 75% accuracy "
        "while getting 0% of real U2R attacks right — every one is "
        "misclassified, most often as harmless 'Normal' traffic. Accuracy "
        "looks fine only because U2R is a tiny slice of the total."
    )
    st.warning(
        "**Finding 2 — the ensemble broke, then got fixed.** A plain "
        "50/50 soft-voting ensemble collapsed to just Naive Bayes's "
        "predictions — 99.2% of NB's probability outputs exceed 0.999 "
        "confidence, drowning out SVM's more realistic estimates in the "
        "average. Weighting SVM's vote higher fixed it, but traded R2L "
        "performance for U2R gains — no single approach wins on every metric."
    )

# =================================================================
# TAB: APPROACH 2
# =================================================================
with tab_a2:
    st.header("Approach 2 — Paper Replication")
    st.caption(
        "Kavitha et al. (IJISAE 2024) methodology, applied to NSL-KDD: "
        "Firefly feature selection, then SVM+NB trained on identical data, "
        "combined via weighted-average voting."
    )

    classifiers = ["Naive Bayes", "SVM", "Ensemble (SVM-NB)"]
    paper_acc = [0.900, 0.910, 0.935]
    our_acc = [0.467, 0.964, 0.966]

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    x = np.arange(len(classifiers))
    width = 0.35
    ax2.bar(x - width / 2, paper_acc, width, label="Paper (KDD99)", color=NEUTRAL)
    ax2.bar(x + width / 2, our_acc, width, label="Our Replication (NSL-KDD)", color=ACCENT)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classifiers)
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Accuracy: paper vs. our NSL-KDD replication")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(paper_acc):
        ax2.text(i - width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(our_acc):
        ax2.text(i + width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    st.pyplot(fig2)

    st.subheader("Firefly-selected features (25 of 41)")
    features = [
        "duration", "service", "land", "urgent", "hot", "num_compromised",
        "root_shell", "su_attempted", "num_file_creations", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "srv_count", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_srv_rerror_rate",
    ]
    st.write(", ".join(f"`{f}`" for f in features))

    st.subheader("Final ensemble — per class")
    df2 = pd.DataFrame({
        "Class": ["Normal", "DoS", "Probe", "R2L", "U2R"],
        "Precision": [0.94, 0.99, 0.98, 0.93, 0.68],
        "Recall": [0.94, 0.97, 0.98, 0.94, 0.62],
        "F1": [0.94, 0.98, 0.98, 0.93, 0.65],
        "Support": [1600, 2000, 2816, 776, 24],
    })
    st.dataframe(df2, use_container_width=True, hide_index=True)

    st.subheader("Confusion matrix — final ensemble")
    labels_order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    cm = np.array([
        [1510, 18, 29, 37, 6],
        [23, 1947, 27, 3, 0],
        [34, 2, 2769, 11, 0],
        [38, 0, 10, 727, 1],
        [3, 0, 0, 6, 15],
    ])
    fig3, ax3 = plt.subplots(figsize=(5.5, 5))
    im = ax3.imshow(cm, cmap="Greens")
    ax3.set_xticks(range(5)); ax3.set_xticklabels(labels_order)
    ax3.set_yticks(range(5)); ax3.set_yticklabels(labels_order)
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
    for i in range(5):
        for j in range(5):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax3.text(j, i, val, ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    st.pyplot(fig3)

    st.warning(
        "**Where we diverge from the paper.** Our SVM and Ensemble "
        "results (96.4% / 96.6%) exceed the paper's (91% / 93.5%) — but "
        "our Naive Bayes result is far lower (46.7% vs 90%). The same "
        "pattern appeared when this was tried on the paper's original "
        "KDD99 dataset too, so it looks systematic rather than a fluke: "
        "the Firefly search uses Naive Bayes as its own fitness function, "
        "which may have converged on a feature subset that favors SVM's "
        "decision boundary far more than NB's independence assumption. "
        "The paper also doesn't report its exact Firefly parameters or "
        "random seed, so an identical reproduction isn't fully possible "
        "from the text alone."
    )

# =================================================================
# TAB: COMPARISON
# =================================================================
with tab_compare:
    st.header("Comparison & Conclusion")

    df3 = pd.DataFrame({
        "Result": [
            "Paper (KDD99, Kavitha et al.)",
            "Approach 1 — best single config",
            "Approach 2 — paper replication (NSL-KDD)",
        ],
        "Accuracy": [0.935, 0.75, 0.966],
        "U2R F1": [None, 0.31, 0.65],
        "Notes": [
            "Reference benchmark, per-class not reported",
            "Weighted ensemble, our own imbalance work",
            "Highest overall, feature-selected",
        ],
    })
    st.dataframe(df3, use_container_width=True, hide_index=True)

    st.success(
        "**Verdict.** Approach 2's feature-selected, same-data ensemble "
        "is the strongest single result — 96.6% accuracy and 0.65 F1 on "
        "U2R, beating the reference paper outright on overall metrics. "
        "Approach 1 is weaker on raw numbers but contributes something "
        "the paper doesn't show at all: a transparent account of *why* "
        "naive ensembling can fail (probability miscalibration) and the "
        "real precision/recall tradeoffs involved in fixing class "
        "imbalance. Together, they satisfy the 'Unified Approach... "
        "using SVM and Naive Bayes' brief while adding a level of "
        "diagnostic honesty beyond reproducing a single headline number."
    )

# =================================================================
# TAB: DEMO (simplified rule-based illustration, NOT the real models)
# =================================================================
with tab_demo:
    st.header("Try the Concept")
    st.caption(
        "A simplified, rule-based illustration of the kind of reasoning a "
        "classifier applies — not the actual trained SVM/Naive Bayes "
        "models, which aren't loaded here. Useful for building intuition "
        "about which traffic patterns signal which attack type."
    )

    col1, col2 = st.columns(2)

    with col1:
        duration = st.slider("Connection duration (seconds)", 0, 400, 2)
        src_bytes = st.slider("Bytes sent by source", 0, 5000, 300)
        count = st.slider("Connections to host (recent count)", 0, 100, 8)
        serror_rate = st.slider("SYN error rate", 0.0, 1.0, 0.05, step=0.05)
        service = st.selectbox("Service", ["http", "ftp", "telnet", "shell", "private"])
        logged_in = st.checkbox("Successfully logged in")
        classify_clicked = st.button("Classify connection", type="primary")

    with col2:
        if classify_clicked:
            if serror_rate > 0.5 and count > 20:
                category, color, reason = "Probe", ALERT, (
                    "A high SYN-error rate combined with many recent "
                    "connections to the same host is a classic port-scanning "
                    "signature — the source is likely probing for open services."
                )
            elif count > 50 and src_bytes < 150:
                category, color, reason = "DoS", ALERT, (
                    "A very high connection count carrying almost no data per "
                    "connection resembles a denial-of-service flood aimed at "
                    "exhausting the host's resources."
                )
            elif not logged_in and service in ("ftp", "telnet") and duration < 5:
                category, color, reason = "R2L", ALERT, (
                    "Repeated brief, failed attempts to reach a remote-access "
                    "service without a successful login resemble a "
                    "remote-to-local unauthorized access attempt."
                )
            elif logged_in and duration > 200 and service == "shell":
                category, color, reason = "U2R", ALERT, (
                    "An unusually long privileged shell session immediately "
                    "after login resembles a user-to-root privilege "
                    "escalation attempt — rare, but high severity."
                )
            else:
                category, color, reason = "Normal", ACCENT, (
                    "Duration, byte count, connection frequency and error "
                    "rate all fall within typical ranges for legitimate "
                    "traffic."
                )

            st.markdown(
                f"<h2 style='color:{color}; margin-bottom:0;'>{category}</h2>",
                unsafe_allow_html=True,
            )
            st.write(reason)
        else:
            st.info("Set the values on the left and click **Classify connection** to see the illustrative prediction.")

st.divider()
st.caption(
    "Built to accompany the NSL-KDD intrusion detection project "
    "(Approach 1 + Approach 2). Figures shown are from actual executed "
    "runs of the project's Python scripts, not placeholders."
)
