# data_cleaning.py

import pandas as pd
import numpy as np
from scipy import stats

def clean_data(df):
    # --- Fill missing values in "Advisory Indication" and adjust "Diagnostic Advisory Indication"
    advisory_cols = [col for col in df.columns if "Advisory Indication" in col and "Diagnostic" not in col]
    diagnostic_cols = [col for col in df.columns if "Diagnostic Advisory Indication" in col]

    if advisory_cols:
        advisory_col = advisory_cols[0]
        df[advisory_col] = df[advisory_col].fillna("Normal").astype(str)

        if diagnostic_cols:
            diagnostic_col = diagnostic_cols[0]
            df[diagnostic_col] = df[diagnostic_col].astype(str)

            # Condition 1: "Normal" → "No Failure Indication"
            condition_normal = df[advisory_col].str.strip().str.lower() == "normal"
            df.loc[
                condition_normal & (df[diagnostic_col].isna() | (df[diagnostic_col].str.strip() == "")),
                diagnostic_col
            ] = "No Failure Indication"

            # Condition 2: High Fired but not Low → Early Failure
            condition_high = df[advisory_col].str.contains("High Fired", case=False, na=False) & \
                             (~df[advisory_col].str.contains("Low Fired", case=False, na=False))
            df.loc[
                condition_high & (df[diagnostic_col].isna() | (df[diagnostic_col].str.strip() == "")),
                diagnostic_col
            ] = "Early Failure Indication - High Sensor Reading Value"

            # Condition 3: Use Very Low Fired diagnostic for Low Fired (if defined)
            low_fired_condition = df[advisory_col].str.contains("Low Fired", case=False, na=False)
            very_low_fired_condition = df[advisory_col].str.contains("Very Low Fired", case=False, na=False)

            very_low_fired_diagnostics = df.loc[very_low_fired_condition, diagnostic_col].dropna().unique()
            if len(very_low_fired_diagnostics) > 0:
                df.loc[
                    low_fired_condition & (df[diagnostic_col].isna() | (df[diagnostic_col].str.strip() == "")),
                    diagnostic_col
                ] = very_low_fired_diagnostics[0]

    # --- Rename mechanical diagnostic labels
    for col in df.columns:
        if "MECH_DIAGNOSTIC" in col:
            df[col] = df[col].replace({
                "Mechanical - Equipment Problem 1": "Mechanical - Equipment Problem - HTC vertical",
                "Mechanical - Equipment Problem 2": "Mechanical - Equipment Problem - HTC axial",
                "Mechanical - Equipment Problem 3": "Mechanical - Equipment Problem - Pump mid side vertical",
                "Mechanical - Equipment Problem 4": "Mechanical - Equipment Problem - Pump mid side horizontal"
            })

    # --- Drop unnecessary columns
    drop_cols = [col for col in df.columns if any(x in col for x in [
        "Smoothed", "Positive Indication", "Negative Indication", "Runtime Indication", "Actual String"
    ])]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # --- Drop empty sensor columns and rows with NaNs
    sensor_columns = [col for col in df.columns if any(x in col for x in ["Actual", "Estimate", "Residual"])]
    empty_columns = [col for col in sensor_columns if df[col].isna().all()]
    if empty_columns:
        df.drop(columns=empty_columns, inplace=True)

    df.dropna(subset=sensor_columns, how="any", inplace=True)

    return df

def remove_duplicates(df, target_col):
    df.drop_duplicates(inplace=True)
    return df

def remove_outliers(df, target_col, verbose=False):
    """Remove outliers using Z-score and IQR on normal cases only. 
    If verbose=True, print before/after shapes."""
    
    sensor_cols = [col for col in df.columns if any(x in col for x in ["Actual", "Estimate", "Residual"])]

    if target_col and target_col in df.columns:
        normal_df = df[df[target_col] == "No Failure Indication"].copy()
        failure_df = df[df[target_col] != "No Failure Indication"].copy()

        if verbose:
            print(f"🧹 Starting outlier removal for normal cases - Initial shape: {normal_df.shape}")

        # Apply Z-score
        if not normal_df.empty:
            z_scores = np.abs(stats.zscore(normal_df[sensor_cols]))
            normal_df = normal_df[(z_scores < 3).all(axis=1)]

            # Apply IQR
            Q1 = normal_df[sensor_cols].quantile(0.25)
            Q3 = normal_df[sensor_cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((normal_df[sensor_cols] < (Q1 - 1.5 * IQR)) | (normal_df[sensor_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            normal_df = normal_df[mask]

            if verbose:
                print(f"✅ Outlier removal completed - Final normal cases shape: {normal_df.shape}")

            # Merge cleaned normal with untouched failure
            df = pd.concat([normal_df, failure_df]).reset_index(drop=True)

    return df

