"""
src/risk_model.py
Stage 1 – Risk Prediction
=========================
Models:
  1. Logistic Regression  – classify risk_label (high/low)
  2. K-Means (k=3)        – cluster into risk_cluster (0=safe, 1=moderate, 2=severe)
  3. KNN (k=3)            – compute risk_score_knn (0–1) from neighbours' historical_flood

Output:
  data/risk_scores.csv   (location_id, risk_label, risk_cluster, risk_score_knn)
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_IN  = ROOT / "data" / "locations.csv"
DATA_OUT = ROOT / "data" / "risk_scores.csv"

# ─── Load data ────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data/locations.csv …")
df = pd.read_csv(DATA_IN)
print(f"  Shape: {df.shape}")

# elevation_m is entirely NaN – drop it from feature consideration
assert df["elevation_m"].isna().all(), "Expected elevation_m to be all-NaN"
print("  elevation_m: all NaN – excluded from features ✓")

# ─── Feature / target prep ───────────────────────────────────────────────────
DIST_COL   = "distance_to_coast_km"
COORD_COLS = ["lat", "lon"]
TARGET     = "historical_flood"          # 0 / 1 binary

X_dist  = df[[DIST_COL]].values         # shape (n, 1)
X_coord = df[COORD_COLS].values         # shape (n, 2)
y       = df[TARGET].values             # 0 / 1

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOGISTIC REGRESSION
#     Feature : distance_to_coast_km
#     Target  : historical_flood (0/1) → risk_label (low/high)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. LOGISTIC REGRESSION")
print("-" * 60)

scaler_lr = StandardScaler()
X_lr = scaler_lr.fit_transform(X_dist)

X_train, X_test, y_train, y_test = train_test_split(
    X_lr, y, test_size=0.2, random_state=42, stratify=y
)

# class_weight='balanced' reweights samples inversely proportional to class
# frequency, preventing the model from collapsing to the majority class (Bug 1 fix)
lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

acc  = accuracy_score(y_test, y_pred_lr)
prec = precision_score(y_test, y_pred_lr, zero_division=0)
rec  = recall_score(y_test, y_pred_lr, zero_division=0)
f1   = f1_score(y_test, y_pred_lr, zero_division=0)
cm   = confusion_matrix(y_test, y_pred_lr)

print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-Score : {f1:.4f}")
print("  Confusion Matrix (rows=actual, cols=predicted):")
print(f"    [TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
print(f"    [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]")

# Predict on full dataset for risk_label
lr_pred_full = lr.predict(X_lr)
df["risk_label"] = np.where(lr_pred_full == 1, "high", "low")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  K-MEANS (k = 3)
#     Feature : distance_to_coast_km
#     Labels  : 0 = safe, 1 = moderate, 2 = severe
#     Clusters ordered by centroid: smallest distance → most severe (2)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. K-MEANS CLUSTERING  (k=3)")
print("-" * 60)

scaler_km = StandardScaler()
X_km = scaler_km.fit_transform(X_dist)

km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_km)

# Map raw cluster ids to semantic labels by centroid distance to coast:
#   smallest centroid → 2 (severe), mid → 1 (moderate), largest → 0 (safe)
centroids_dist = scaler_km.inverse_transform(km.cluster_centers_).flatten()
order = np.argsort(centroids_dist)   # [idx_smallest, idx_mid, idx_largest]
severity_rank = {int(raw_id): (2 - rank) for rank, raw_id in enumerate(order)}

df["risk_cluster"] = np.vectorize(severity_rank.get)(km.labels_)

sil = silhouette_score(X_km, km.labels_)
print(f"  Silhouette Score: {sil:.4f}")
print("  Cluster centroids (distance_to_coast_km):")
for raw_id, sem in sorted(severity_rank.items(), key=lambda x: x[1], reverse=True):
    label_name = {2: "severe", 1: "moderate", 0: "safe"}[sem]
    print(f"    raw cluster {raw_id} → {sem} ({label_name})  "
          f"centroid = {centroids_dist[raw_id]:.2f} km")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  KNN (k = 3)
#     Similarity basis : lat / lon (geographic proximity)
#     risk_score_knn   : fraction of 3 nearest neighbours with historical_flood=1
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. KNN RISK SCORE  (k=3, features=lat/lon)")
print("-" * 60)

scaler_knn = StandardScaler()
X_knn = scaler_knn.fit_transform(X_coord)

# Use NearestNeighbors with n=4 so that, after dropping the self-match
# (index column 0, distance 0), exactly 3 external neighbours remain.
# This eliminates the self-inclusion leakage present when KNeighborsRegressor
# fits and predicts on the same dataset (Bug 2 fix).
nn = NearestNeighbors(n_neighbors=4, metric="euclidean")
nn.fit(X_knn)
_, indices = nn.kneighbors(X_knn)
neighbor_idx = indices[:, 1:]                          # drop col 0 (self)
risk_score_knn = y[neighbor_idx].astype(float).mean(axis=1)
df["risk_score_knn"] = np.round(risk_score_knn, 4)

print(f"  risk_score_knn range : [{df['risk_score_knn'].min():.4f}, "
      f"{df['risk_score_knn'].max():.4f}]")
print(f"  Mean risk_score_knn  : {df['risk_score_knn'].mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SAVE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. SAVING  data/risk_scores.csv")
print("-" * 60)

out = df[["location_id", "risk_label", "risk_cluster", "risk_score_knn"]].copy()
out.to_csv(DATA_OUT, index=False)
print(f"  Saved {len(out)} rows → {DATA_OUT}")

# ─── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY TABLE  (first 10 rows)")
print("-" * 60)
print(out.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("VALUE COUNTS – risk_label")
print("-" * 60)
print(out["risk_label"].value_counts().to_string())

print("\nVALUE COUNTS – risk_cluster")
print("-" * 60)
cluster_label_map = {0: "safe", 1: "moderate", 2: "severe"}
for cluster_id, count in out["risk_cluster"].value_counts().sort_index().items():
    print(f"  {cluster_id} ({cluster_label_map[cluster_id]:>8}): {count}")

print("\n✅  Stage 1 complete.")
