"""
Clinical Decision Support System (CDSS) — ML Training Pipeline
Type 2 Diabetes Drug Effectiveness Prediction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(ROOT, "data", "diabetes_drug_effectiveness.csv")
MODELS_DIR  = os.path.join(ROOT, "models")
PLOTS_DIR   = os.path.join(MODELS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Feature sets ───────────────────────────────────────────────────────────
BASELINE_FEATURES = [
    "age", "gender", "bmi",
    "hba1c_v1", "glucose_v1", "bp_v1",
    "metformin_v1", "sulfonylurea_v1", "dpp4_v1", "sglt2_v1", "insulin_v1",
    "adherence", "diet", "exercise"
]
TARGET = "treatment_effectiveness"

# ══════════════════════════════════════════════════════════════════════════
# 1.  LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("CDSS — Loading and preprocessing data …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Target distribution:\n{df[TARGET].value_counts()}")

X = df[BASELINE_FEATURES].copy()
y = df[TARGET].copy()

# train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE — train size: {X_train_sm.shape[0]}")

# Keep a classifier scaler artifact separate from clustering. The tree
# classifiers train on raw values, but retaining this artifact preserves the
# existing model bundle without coupling it to KMeans preprocessing.
classifier_scaler = StandardScaler()
classifier_scaler.fit(X_train_sm)
joblib.dump(classifier_scaler, f"{MODELS_DIR}/scaler.pkl")

# ══════════════════════════════════════════════════════════════════════════
# 2.  TRAIN CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=42
    ),
}

results = {}
best_auc = -1
best_name = None

for name, clf in models.items():
    print(f"\n── Training {name} …")
    clf.fit(X_train_sm, y_train_sm)

    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm
    }
    print(f"  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

    joblib.dump(clf, f"{MODELS_DIR}/{name}.pkl")

    if auc > best_auc:
        best_auc  = auc
        best_name = name

print(f"\n  Best model: {best_name} (AUC={best_auc:.4f})")
joblib.dump(models[best_name], f"{MODELS_DIR}/best_model.pkl")

# ── Feature importance ────────────────────────────────────────────────────
best_clf = models[best_name]
if hasattr(best_clf, "feature_importances_"):
    fi = pd.Series(best_clf.feature_importances_, index=BASELINE_FEATURES).sort_values(ascending=False)
    fi_dict = fi.to_dict()
else:
    fi_dict = {}

# ══════════════════════════════════════════════════════════════════════════
# 3.  CLUSTERING (KMeans)
# ══════════════════════════════════════════════════════════════════════════
print("\n── KMeans clustering …")
cluster_features = ["age", "bmi", "hba1c_v1", "glucose_v1", "bp_v1", "adherence"]
cluster_scaler = StandardScaler()
X_cluster = cluster_scaler.fit_transform(df[cluster_features])
joblib.dump(cluster_scaler, f"{MODELS_DIR}/cluster_scaler.pkl")

# Elbow
inertias = []
K_range  = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

optimal_k = 4
km_final  = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X_cluster)
joblib.dump(km_final, f"{MODELS_DIR}/kmeans.pkl")

cluster_summary = {}
for c in range(optimal_k):
    sub = df[df["cluster"] == c]
    cluster_summary[int(c)] = {
        "size": int(len(sub)),
        "mean_hba1c": round(float(sub["hba1c_v1"].mean()), 2),
        "mean_bmi":   round(float(sub["bmi"].mean()), 2),
        "mean_age":   round(float(sub["age"].mean()), 2),
        "mean_adherence": round(float(sub["adherence"].mean()), 2),
        "effectiveness_rate": round(float((sub[TARGET] == 0).mean()), 3),
        "dominant_drugs": {
            "metformin":    round(float(sub["metformin_v1"].mean()), 2),
            "sulfonylurea": round(float(sub["sulfonylurea_v1"].mean()), 2),
            "dpp4":         round(float(sub["dpp4_v1"].mean()), 2),
            "sglt2":        round(float(sub["sglt2_v1"].mean()), 2),
            "insulin":      round(float(sub["insulin_v1"].mean()), 2),
        }
    }
print(f"  Cluster sizes: {[cluster_summary[c]['size'] for c in range(optimal_k)]}")

# ══════════════════════════════════════════════════════════════════════════
# 4.  GENERATE PLOTS
# ══════════════════════════════════════════════════════════════════════════
print("\n── Generating plots …")

plt.style.use("seaborn-v0_8-whitegrid")
colors = ["#0ea5e9", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]

# — Feature importance
fig, ax = plt.subplots(figsize=(10, 6))
fi_s = pd.Series(fi_dict).sort_values()
fi_s.plot(kind="barh", ax=ax, color="#0ea5e9", edgecolor="white")
ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# — Correlation heatmap (baseline features + target)
corr_cols = BASELINE_FEATURES + [TARGET]
fig, ax = plt.subplots(figsize=(12, 9))
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, annot_kws={"size": 8})
ax.set_title("Correlation Heatmap — Baseline Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# — HbA1c trends v1→v3
fig, ax = plt.subplots(figsize=(10, 5))
for label, color in zip([0, 1], ["#10b981", "#ef4444"]):
    sub = df[df[TARGET] == label]
    means = [sub["hba1c_v1"].mean(), sub["hba1c_v2"].mean(), sub["hba1c_v3"].mean()]
    ax.plot(["Visit 1", "Visit 2", "Visit 3"], means, marker="o",
            label="Effective" if label == 0 else "Not Effective", color=color, linewidth=2.5)
ax.set_title("HbA1c Trend by Treatment Outcome", fontsize=14, fontweight="bold")
ax.set_ylabel("Mean HbA1c (%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/hba1c_trend.png", dpi=150, bbox_inches="tight")
plt.close()

# — Cluster distribution
fig, ax = plt.subplots(figsize=(8, 5))
cluster_sizes = [cluster_summary[c]["size"] for c in range(optimal_k)]
cluster_labels = [f"Cluster {c}" for c in range(optimal_k)]
ax.bar(cluster_labels, cluster_sizes, color=colors[:optimal_k], edgecolor="white", linewidth=1.5)
ax.set_title("Patient Cluster Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Patients")
for i, v in enumerate(cluster_sizes):
    ax.text(i, v + 10, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/cluster_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# — ROC Curve (both models)
fig, ax = plt.subplots(figsize=(8, 6))
for name, clf in models.items():
    y_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = results[name]["roc_auc"]
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# — Confusion matrix for best model
fig, ax = plt.subplots(figsize=(6, 5))
cm = np.array(results[best_name]["confusion_matrix"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Effective", "Not Effective"],
            yticklabels=["Effective", "Not Effective"])
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# — Elbow curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(K_range), inertias, marker="o", color="#0ea5e9", linewidth=2.5)
ax.axvline(optimal_k, linestyle="--", color="#ef4444", label=f"Optimal k={optimal_k}")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
ax.set_title("KMeans Elbow Method", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/elbow_curve.png", dpi=150, bbox_inches="tight")
plt.close()

print("  Plots saved.")

# ══════════════════════════════════════════════════════════════════════════
# 5.  SAVE METADATA
# ══════════════════════════════════════════════════════════════════════════
metadata = {
    "best_model": best_name,
    "target_encoding": {"0": "Effective", "1": "Not Effective"},
    "features": BASELINE_FEATURES,
    "results": results,
    "feature_importance": {k: round(v, 4) for k, v in fi_dict.items()},
    "cluster_summary": cluster_summary,
    "optimal_k": optimal_k,
    "cluster_features": cluster_features,
    "dataset_size": int(len(df)),
    "legacy_artifacts": {"scaler.pkl": "Legacy classifier scaler retained for compatibility; predictive inference uses raw features and does not load this artifact."},
    "class_distribution": df[TARGET].value_counts().to_dict()
}
with open(f"{MODELS_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Training complete. All artifacts saved to models/")
print(json.dumps(results, indent=2))
