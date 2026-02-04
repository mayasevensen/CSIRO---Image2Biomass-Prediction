import pandas as pd
import numpy as np


# Canonical target order (use this everywhere)
TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
WEIGHTS = {"Dry_Green_g": 0.1, "Dry_Dead_g": 0.1, "Dry_Clover_g": 0.1, "GDM_g": 0.2, "Dry_Total_g": 0.5}


def load_train_data(csv_path):
    """
    Loads training data and pivots it so 1 row = 1 image.
    Returns: 
        df_wide: DataFrame with one row per image.
        y_values: Numpy array of shape (N_images, 5) containing the targets.
    """
    df = pd.read_csv(csv_path)
    
    # Extract image_id from sample_id (e.g., "ID123__Total" -> "ID123")
    df["image_id"] = df["sample_id"].str.split("__").str[0]
    
    # Pivot to wide format (1 row per image, targets as columns)
    df_wide = df.pivot_table(
        index=["image_id", "image_path"],
        columns="target_name",
        values="target"
    ).reset_index()
    
    # Get the target values in the specific order
    y_values = df_wide[TARGETS].values
    
    return df_wide, y_values

def prepare_submission(test_csv_path, predictions, image_ids):
    """
    Maps the 5-value predictions back to the long-format test CSV rows.
    
    Args:
        test_csv_path: Path to original test.csv
        predictions: Numpy array of shape (N_test_images, 5)
        image_ids: List of image IDs corresponding to the predictions
    """
    df_test = pd.read_csv(test_csv_path)
    
    # Create a lookup dictionary: {image_id: {target_name: value}}
    pred_dict = {}
    for img_id, pred_vector in zip(image_ids, predictions):
        pred_dict[img_id] = {
            col: val for col, val in zip(TARGETS, pred_vector)
        }
    
    # Helper function to look up the value for each row
    def get_pred(row):
        # Extract ID from "ID123__Dry_Total_g"
        img_id = row['sample_id'].split('__')[0]
        target_name = row['target_name']
        
        # specific lookup
        val = pred_dict.get(img_id, {}).get(target_name, 0.0)
        
        # Safety: Biomass cannot be negative
        return max(0.0, val)
    
    df_test['target'] = df_test.apply(get_pred, axis=1)
    
    return df_test[['sample_id', 'target']]


from sklearn.metrics import mean_squared_error

def weighted_global_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Kaggle metric: one global weighted R^2 across all (image, target) pairs.
    y_true/y_pred shape: (N, 5) in TARGETS order.
    """
    assert y_true.shape == y_pred.shape and y_true.shape[1] == 5
    w = np.array([WEIGHTS[t] for t in TARGETS], dtype=np.float64)  # (5,)

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


def time_split(df: pd.DataFrame, val_frac: float = 0.2):
    df = df.copy()
    df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"])
    df = df.sort_values("Sampling_Date").reset_index(drop=True)
    n_val = int(len(df) * val_frac)
    train_df = df.iloc[:-n_val].reset_index(drop=True)
    val_df   = df.iloc[-n_val:].reset_index(drop=True)
    return train_df, val_df
