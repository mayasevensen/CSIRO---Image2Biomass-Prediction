import pandas as pd
import numpy as np

# The 5 specific targets you need to predict
TARGET_COLS = ['Dry_Total_g', 'Dry_Green_g', 'Dry_Dead_g', 'GDM_g', 'Dry_Clover_g']

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
    y_values = df_wide[TARGET_COLS].values
    
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
            col: val for col, val in zip(TARGET_COLS, pred_vector)
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