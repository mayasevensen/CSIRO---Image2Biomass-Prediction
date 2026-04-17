"""
PyTorch Dataset for the CEMS pipeline.

Reads from cached DINOv2 features (no per-epoch DINOv2 inference).
Train/val split uses GroupKFold on image_path (n_splits=5, fold 0 as val)
to prevent leakage from multi-row images in the long-format CSV.
Logic copied from pipeline_CHARMS.ipynb.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CSV_PATH = REPO_ROOT / "data" / "tabular" / "train" / "train.csv"

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _make_wide_df(csv_path):
    """One row per image, with 5 target columns. Copied from pipeline_CHARMS."""
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


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def make_splits(csv_path=CSV_PATH, n_splits=5, val_fold=0):
    """
    Returns (train_ids, val_ids): lists of image ID strings that index into
    the cached feature arrays.
    """
    df = _make_wide_df(csv_path)
    image_ids_csv = df["image_path"].apply(_image_id_from_path).values

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(df, groups=df["image_path"]))
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
    Returns (train_dataset, val_dataset, label_mean, label_std).
    Features are NOT normalised (frozen DINOv2 CLS tokens; downstream MLP
    can handle this, and standardising would distort ID structure).
    Labels are returned raw — the loss function uses raw scales.
    """
    # Load cache
    features = np.load(cache_dir / "features_dinov2.npy")   # (N, 384)
    image_ids = np.load(cache_dir / "image_ids.npy", allow_pickle=True)  # (N,)

    # Build label array aligned to cache order
    df = _make_wide_df(csv_path)
    id_to_label = {
        _image_id_from_path(row["image_path"]): row[TARGETS].values.astype(np.float32)
        for _, row in df.iterrows()
    }
    labels = np.stack([id_to_label.get(iid, np.zeros(5, dtype=np.float32))
                       for iid in image_ids])  # (N, 5)

    id_to_idx = {iid: i for i, iid in enumerate(image_ids)}

    train_ids, val_ids = make_splits(csv_path)

    train_ds = BiomassFeatureDataset(train_ids, features, labels, id_to_idx)
    val_ds = BiomassFeatureDataset(val_ids, features, labels, id_to_idx)

    return train_ds, val_ds
