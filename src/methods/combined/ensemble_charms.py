"""Blend combined_pipeline's submission with CHARMS's submission.

Usage:
    python ensemble_charms.py \
        --combined_submission submission_combined.csv \
        --charms_submission   submission_charms.csv  \
        --combined_oof        oof_combined.npz       \
        --charms_oof          oof_charms.npz         \
        --out                 submission_ensembled.csv

If either `--combined_oof` or `--charms_oof` is missing, falls back to w=0.5.

Competition metric: weighted global R² over all (image, target) pairs with
target weights 0.1/0.1/0.1/0.2/0.5 for Dry_Green_g / Dry_Dead_g / Dry_Clover_g
/ GDM_g / Dry_Total_g.

OOF files are expected to be `np.savez` archives with at least:
  - oof_preds : (N, 5) float32 — per-target predictions for rows in oof_mask
  - oof_mask  : (N,)   bool    — which rows are covered
  - y_true    : (N, 5) float32 — ground-truth targets
  - real_image_ids : (N,) str  — identifier per row (repeated for flip4x rows)

The OOF may include flip4x-duplicated rows; we simply concatenate all valid
rows in both files and assume row order matches for the two models. (Both
pipelines use identical GroupKFold(n_splits=5, seed=42).)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
WEIGHT_VECTOR = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)


def weighted_global_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    ww = np.repeat(WEIGHT_VECTOR, y_true.shape[0])
    ybar   = np.sum(ww * yt) / np.sum(ww)
    ss_res = np.sum(ww * (yt - yp) ** 2)
    ss_tot = np.sum(ww * (yt - ybar) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def load_oof(path: str):
    """Return (preds, true) filtered by oof_mask, or (None, None) if unavailable."""
    if not path or not os.path.exists(path):
        return None, None
    data = np.load(path, allow_pickle=True)
    if not {"oof_preds", "oof_mask", "y_true"}.issubset(set(data.files)):
        print(f"  OOF file {path} missing required arrays {data.files} — skipping.")
        return None, None
    mask = data["oof_mask"].astype(bool)
    return data["oof_preds"][mask], data["y_true"][mask]


def find_optimal_weight(p_combined, p_charms, y_true, n_grid=101):
    """Grid search w in [0, 1] for max weighted R²."""
    best_w, best_r2 = 0.5, -float("inf")
    for w in np.linspace(0.0, 1.0, n_grid):
        blend = w * p_combined + (1.0 - w) * p_charms
        r2    = weighted_global_r2(y_true, blend)
        if r2 > best_r2:
            best_r2, best_w = r2, float(w)
    return best_w, best_r2


def blend_submissions(combined_csv: str, charms_csv: str, w: float) -> pd.DataFrame:
    df_c = pd.read_csv(combined_csv)
    df_h = pd.read_csv(charms_csv)
    assert len(df_c) == len(df_h), (
        f"Row count mismatch: combined={len(df_c)}  charms={len(df_h)}"
    )
    # Align on sample_id to be safe
    df_m = df_c.merge(df_h, on="sample_id", suffixes=("_c", "_h"))
    assert len(df_m) == len(df_c), "sample_id mismatch between CSVs"
    df_m["target"] = w * df_m["target_c"] + (1.0 - w) * df_m["target_h"]
    df_m["target"] = df_m["target"].clip(lower=0.0)
    return df_m[["sample_id", "target"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined_submission", required=True)
    p.add_argument("--charms_submission",   required=True)
    p.add_argument("--combined_oof", default=None)
    p.add_argument("--charms_oof",   default=None)
    p.add_argument("--out", default="submission_ensembled.csv")
    p.add_argument("--weight", type=float, default=None,
                   help="Force w (overrides OOF grid search).")
    args = p.parse_args()

    if args.weight is not None:
        w, r2 = float(args.weight), float("nan")
        print(f"Using forced weight w = {w:.3f}")
    else:
        pc, yc = load_oof(args.combined_oof)
        ph, yh = load_oof(args.charms_oof)
        if pc is None or ph is None:
            print("OOF unavailable for one or both models — defaulting to w=0.5.")
            w, r2 = 0.5, float("nan")
        else:
            if pc.shape != ph.shape:
                print(
                    f"OOF shape mismatch: combined {pc.shape} vs charms {ph.shape} — "
                    "defaulting to w=0.5."
                )
                w, r2 = 0.5, float("nan")
            elif not np.allclose(yc, yh):
                print("OOF y_true mismatch between files — defaulting to w=0.5.")
                w, r2 = 0.5, float("nan")
            else:
                w, r2 = find_optimal_weight(pc, ph, yc)
                print(f"Grid search: best w = {w:.3f}  OOF weighted R² = {r2:.4f}")

    out = blend_submissions(args.combined_submission, args.charms_submission, w)
    out.to_csv(args.out, index=False)
    print(f"Wrote ensembled submission to {args.out}  (rows={len(out)}, w={w:.3f})")


if __name__ == "__main__":
    main()
