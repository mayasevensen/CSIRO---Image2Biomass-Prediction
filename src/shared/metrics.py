"""
Metric functions for the CEMS pipeline.
Copied from src/models/data_utils.py — no cross-folder imports.

Includes a small synthetic verification to catch regressions.
"""

import numpy as np
from sklearn.metrics import mean_squared_error

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}
WEIGHT_VECTOR = np.array([WEIGHTS[t] for t in TARGETS], dtype=np.float64)


def weighted_global_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Kaggle metric: one global weighted R² across all (image, target) pairs.
    y_true / y_pred shape: (N, 5) in TARGETS order.
    """
    assert y_true.shape == y_pred.shape and y_true.shape[1] == 5
    w = WEIGHT_VECTOR  # (5,)

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    ww = np.repeat(w, y_true.shape[0])  # (N*5,)

    ybar = np.sum(ww * yt) / np.sum(ww)
    ss_res = np.sum(ww * (yt - yp) ** 2)
    ss_tot = np.sum(ww * (yt - ybar) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def rmse_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for i, t in enumerate(TARGETS):
        out[t] = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
    return out


# ---------------------------------------------------------------------------
# Sanity check with known answers
# ---------------------------------------------------------------------------

def _verify_metrics():
    """
    Perfect predictions must give R²=1.0; predicting the weighted mean gives R²=0.0.
    Verified analytically.
    """
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 100, size=(50, 5))

    # Perfect prediction
    r2_perfect = weighted_global_r2(y, y)
    assert abs(r2_perfect - 1.0) < 1e-9, f"Perfect prediction R²={r2_perfect} ≠ 1"

    # Predict the weighted mean constant — R² should be 0
    w = WEIGHT_VECTOR
    yt = y.reshape(-1)
    ww = np.repeat(w, y.shape[0])
    ybar = np.sum(ww * yt) / np.sum(ww)
    y_mean = np.full_like(y, ybar)
    r2_mean = weighted_global_r2(y, y_mean)
    assert abs(r2_mean) < 1e-6, f"Mean-prediction R²={r2_mean} ≠ 0"

    # RMSE: identical arrays → 0
    rmse = rmse_per_target(y, y)
    for t in TARGETS:
        assert rmse[t] == 0.0, f"Zero-error RMSE for {t} = {rmse[t]} ≠ 0"

    print("metrics.py: all sanity checks passed.")


if __name__ == "__main__":
    _verify_metrics()
