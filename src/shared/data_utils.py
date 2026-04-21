"""
Data utilities for the Image2Biomass pipeline.

Covers:
- Path constants (repo-relative)
- Target definitions and weights
- Wide-format DataFrame helpers (_make_wide_df, load_train_data)
- GroupKFold split builder (make_splits)
- PyTorch Dataset (BiomassFeatureDataset, load_datasets)
- Submission helpers (prepare_submission)
- Metric functions (weighted_global_r2, rmse_per_target)
- Time-based split (time_split)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
CSV_PATH = REPO_ROOT / "data" / "tabular" / "train" / "train.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
WEIGHTS = {"Dry_Green_g": 0.1, "Dry_Dead_g": 0.1, "Dry_Clover_g": 0.1, "GDM_g": 0.2, "Dry_Total_g": 0.5}


# ---------------------------------------------------------------------------
# Wide-format helpers
# ---------------------------------------------------------------------------

def _make_wide_df(csv_path):
    """One row per image, with 5 target columns."""
    df = pd.read_csv(csv_path)
    meta_cols = ["sample_id", "image_path", "Sampling_Date",
                 "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"]
    meta = df[meta_cols].drop_duplicates("image_path").copy()
    y = (
        df.pivot_table(
            index="image_path",
            columns="target_name",
            values="target",
            aggfunc="first",
        )
        .reset_index()
    )
    return meta.merge(y, on="image_path", how="inner")


def _image_id_from_path(image_path_str):
    """'train/ID123.jpg'  →  'ID123'"""
    return Path(image_path_str).stem


def load_train_data(csv_path):
    """
    Loads training data and pivots it so 1 row = 1 image.
    Returns (df_wide, y_values) where y_values has shape (N_images, 5).
    """
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
# Split builder
# ---------------------------------------------------------------------------

def _visit_group_key(df: pd.DataFrame) -> pd.Series:
    """
    A paddock-visit key: Sampling_Date + State + Species.

    The CSV has 357 images but only ~40 unique (date, state, species) combos,
    so many images come from the same visit. Grouping on image_path (the old
    default) is effectively no-op — GroupKFold degenerates into KFold and
    images from the same visit leak across splits.
    """
    return (
        pd.to_datetime(df["Sampling_Date"]).dt.strftime("%Y-%m-%d")
        + "__" + df["State"].astype(str)
        + "__" + df["Species"].astype(str)
    )


def make_splits(csv_path=CSV_PATH, n_splits=5, val_fold=0, group_by="visit"):
    """
    Returns (train_ids, val_ids): lists of image ID strings that index into
    the cached feature arrays.

    Args:
        group_by: "visit" (default) — group by Sampling_Date + State + Species.
                  "image_path" — legacy behaviour (groups are unique per image,
                  so GroupKFold ≡ KFold; kept only for reproducibility of
                  older results).
    """
    df = _make_wide_df(csv_path)
    image_ids_csv = df["image_path"].apply(_image_id_from_path).values

    if group_by == "visit":
        groups = _visit_group_key(df).values
    elif group_by == "image_path":
        groups = df["image_path"].values
    else:
        raise ValueError(f"Unknown group_by={group_by!r}")

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(df, groups=groups))
    train_idx, val_idx = splits[val_fold]

    train_ids = image_ids_csv[train_idx].tolist()
    val_ids = image_ids_csv[val_idx].tolist()
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BiomassFeatureDataset(Dataset):
    """
    Loads pre-extracted DINOv2 features and labels.

    Args:
        ids: list of image ID strings (e.g. ['ID123', 'ID456', ...])
        features: (N_total, 384) numpy array (full cache)
        labels:   (N_total, 5)   numpy array (full cache, aligned)
        id_to_idx: dict mapping image_id → row index in the full arrays
    """

    def __init__(self, ids, features, labels, id_to_idx):
        indices = [id_to_idx[i] for i in ids if i in id_to_idx]
        self.X = torch.tensor(features[indices], dtype=torch.float32)
        self.y = torch.tensor(labels[indices], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Top-level factory
# ---------------------------------------------------------------------------

def load_datasets(csv_path=CSV_PATH, cache_dir=CACHE_DIR):
    """
    Returns (train_dataset, val_dataset).
    Features are NOT normalised (frozen DINOv2 CLS tokens).
    Labels are returned raw — the loss function uses raw scales.
    """
    features = np.load(cache_dir / "features_dinov2.npy")
    image_ids = np.load(cache_dir / "image_ids.npy", allow_pickle=True)

    df = _make_wide_df(csv_path)
    id_to_label = {
        _image_id_from_path(row["image_path"]): row[TARGETS].values.astype(np.float32)
        for _, row in df.iterrows()
    }

    # Label coverage: any cached image_id used in a split MUST have a label.
    # Previously we silently substituted a zero vector for missing IDs, which
    # could corrupt training.
    train_ids, val_ids = make_splits(csv_path)
    missing = [iid for iid in list(train_ids) + list(val_ids)
               if iid not in id_to_label]
    assert not missing, (
        f"{len(missing)} split IDs have no label in the CSV "
        f"(first few: {missing[:5]})"
    )

    labels = np.stack([id_to_label.get(iid, np.zeros(5, dtype=np.float32))
                       for iid in image_ids])

    id_to_idx = {iid: i for i, iid in enumerate(image_ids)}

    train_ds = BiomassFeatureDataset(train_ids, features, labels, id_to_idx)
    val_ds = BiomassFeatureDataset(val_ids, features, labels, id_to_idx)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Submission helpers
# ---------------------------------------------------------------------------

def prepare_submission(test_csv_path, predictions, image_ids):
    """
    Maps the 5-value predictions back to the long-format test CSV rows.

    Args:
        test_csv_path: Path to original test.csv
        predictions: Numpy array of shape (N_test_images, 5)
        image_ids: List of image IDs corresponding to the predictions
    """
    df_test = pd.read_csv(test_csv_path)

    pred_dict = {}
    for img_id, pred_vector in zip(image_ids, predictions):
        pred_dict[img_id] = {col: val for col, val in zip(TARGETS, pred_vector)}

    def get_pred(row):
        img_id = row['sample_id'].split('__')[0]
        target_name = row['target_name']
        val = pred_dict.get(img_id, {}).get(target_name, 0.0)
        return max(0.0, val)

    df_test['target'] = df_test.apply(get_pred, axis=1)
    return df_test[['sample_id', 'target']]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def weighted_global_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Kaggle metric: one global weighted R^2 across all (image, target) pairs.
    y_true/y_pred shape: (N, 5) in TARGETS order.
    """
    assert y_true.shape == y_pred.shape and y_true.shape[1] == 5
    w = np.array([WEIGHTS[t] for t in TARGETS], dtype=np.float64)

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    ww = np.repeat(w, y_true.shape[0])

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
# Time-based split
# ---------------------------------------------------------------------------

def time_split(df: pd.DataFrame, val_frac: float = 0.2):
    df = df.copy()
    df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"])
    df = df.sort_values("Sampling_Date").reset_index(drop=True)
    n_val = int(len(df) * val_frac)
    train_df = df.iloc[:-n_val].reset_index(drop=True)
    val_df   = df.iloc[-n_val:].reset_index(drop=True)
    return train_df, val_df
