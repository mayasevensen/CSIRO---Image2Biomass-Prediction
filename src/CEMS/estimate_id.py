"""
Intrinsic dimension (ID) estimation for CEMS Part 1.

Estimates ID on:
  (a) Raw DINOv2 features  (N, 384)
  (b) Raw ResNet50 features (N, 2048)
  (c) Labels alone          (N, 5)

Uses TwoNN estimator (scikit-dimension if available, in-house fallback).
Label loading logic copied from src/models/data_utils.py — no cross-folder imports.
"""

import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CSV_PATH = REPO_ROOT / "data" / "tabular" / "train" / "train.csv"

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

# ---------------------------------------------------------------------------
# Label loading  (copied from data_utils.load_train_data)
# ---------------------------------------------------------------------------

def load_labels(csv_path):
    """Return (df_wide, y_values (N,5)) with one row per image."""
    df = pd.read_csv(csv_path)
    df["image_id"] = df["sample_id"].str.split("__").str[0]
    df_wide = df.pivot_table(
        index=["image_id", "image_path"],
        columns="target_name",
        values="target",
    ).reset_index()
    y_values = df_wide[TARGETS].values
    return df_wide, y_values


# ---------------------------------------------------------------------------
# TwoNN: scikit-dimension or in-house fallback
# ---------------------------------------------------------------------------

def _twonn_skdim(X):
    from skdim.id import TwoNN
    return float(TwoNN().fit(X).dimension_)


def _twonn_inhouse(X):
    """
    In-house TwoNN (Facco et al. 2017).
      1. For each point find r1 (nearest) and r2 (2nd-nearest) distances.
      2. μ_i = r2_i / r1_i.
      3. Empirical survival: P(μ > t) = exp(−d·log(t))  ⇒  linear regression.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto", metric="euclidean")
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    r1 = dists[:, 1]  # nearest (col 0 is self)
    r2 = dists[:, 2]  # second nearest
    # Drop degenerate points where r1 == 0
    mask = r1 > 0
    r1, r2 = r1[mask], r2[mask]
    mu = r2 / r1
    mu_sorted = np.sort(mu)
    n = len(mu_sorted)
    ecdf = np.arange(1, n + 1) / n
    # −log(1 − F(μ)) = d · log(μ)  →  fit via lstsq (no intercept)
    log_mu = np.log(mu_sorted)
    log_surv = -np.log1p(-ecdf + 1e-10)
    d = float(np.dot(log_mu, log_surv) / np.dot(log_mu, log_mu))
    return d


def estimate_id(X, label=""):
    """Try scikit-dimension, fall back to in-house TwoNN."""
    try:
        import skdim  # noqa: F401
    except ImportError:
        print("  scikit-dimension not found — attempting pip install...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "scikit-dimension", "-q"]
            )
        except Exception:
            pass

    try:
        d = _twonn_skdim(X)
        method = "skdim.TwoNN"
    except Exception:
        d = _twonn_inhouse(X)
        method = "in-house TwoNN"

    if label:
        print(f"  [{label}]  d = {d:.2f}  (method: {method})")
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- load features ----
    print("Loading cached features...")
    feats_dino = np.load(CACHE_DIR / "features_dinov2.npy")
    feats_resnet = np.load(CACHE_DIR / "features_resnet50.npy")
    image_ids_cached = np.load(CACHE_DIR / "image_ids.npy", allow_pickle=True)
    N = len(feats_dino)
    print(f"  DINOv2:  {feats_dino.shape}")
    print(f"  ResNet50:{feats_resnet.shape}")
    print(f"  N = {N}")

    # ---- load labels and align to cached image order ----
    print("\nLoading labels...")
    df_wide, y_all = load_labels(CSV_PATH)
    # align label rows to the order in the cache
    id_to_idx = {row["image_id"]: i for i, row in df_wide.iterrows()}
    label_order = [id_to_idx[img_id] for img_id in image_ids_cached
                   if img_id in id_to_idx]
    y_values = y_all[label_order]
    print(f"  Labels shape: {y_values.shape}")

    # ---- ID estimation ----
    print("\nEstimating intrinsic dimension (TwoNN)...")
    d_dino   = estimate_id(feats_dino,   label="DINOv2 384-d")
    d_resnet = estimate_id(feats_resnet, label="ResNet50 2048-d")
    d_labels = estimate_id(y_values,     label="Labels 5-d")

    # ---- print summary ----
    print("\n" + "=" * 55)
    print("PART 1 — Intrinsic Dimension Summary")
    print("=" * 55)
    print(f"  N (training images)         : {N}")
    print(f"  (a) DINOv2 raw features     : d = {d_dino:.2f}")
    print(f"  (b) ResNet50 raw features   : d = {d_resnet:.2f}")
    print(f"  (c) Labels only (5-d)       : d = {d_labels:.2f}")
    print()
    lower_encoder = "DINOv2" if d_dino <= d_resnet else "ResNet50"
    print(f"  Encoder with lower raw d    : {lower_encoder}")
    print()
    print("  d² vs batch sizes (using DINOv2 d):")
    print(f"    d²  = {d_dino**2:.1f}")
    for bs in [16, 32, 64]:
        verdict = "OK" if bs >= d_dino ** 2 else "TIGHT (<d²)"
        print(f"    bs={bs:3d}  {'≥' if bs >= d_dino**2 else '<'} d²  → {verdict}")
    print()
    print("  NOTE: The decisive feasibility estimate is on the *learned*")
    print("  32-d latent (Part 2). These raw-feature IDs are a sanity")
    print("  check on the encoder choice, not the final CEMS verdict.")
    print("=" * 55)

    return {
        "N": N,
        "d_dino": d_dino,
        "d_resnet": d_resnet,
        "d_labels": d_labels,
    }


if __name__ == "__main__":
    main()
