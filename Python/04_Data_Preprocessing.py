# =============================================================================
# Project : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# File    : 04_Data_Preprocessing.py
# Phase   : Data Preprocessing
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# Locate Project Root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# =============================================================================
# Import Configuration
# =============================================================================

from config import (
    WORKING_DATA_FILE,
    CLEAN_DATA_FOLDER,
    RESULTS_FOLDER,
    TARGET_COLUMN,
    TARGET_MAPPING
)

# =============================================================================
# Display Settings
# =============================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# =============================================================================
# STEP 1 : LOAD DATASET
# =============================================================================

print("=" * 70)
print("STEP 1 : LOADING DATASET")
print("=" * 70)

df = pd.read_excel(WORKING_DATA_FILE)

print("Dataset loaded successfully.")

# =============================================================================
# STEP 2 : ENCODE TARGET VARIABLE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2 : ENCODING TARGET VARIABLE")
print("=" * 70)

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .str.lower()
    .str.strip()
    .map(TARGET_MAPPING)
)

encoded_target = pd.DataFrame({
    "Original Label": ["alive", "failed"],
    "Encoded Value": [0, 1]
})

print(encoded_target)

# =============================================================================
# STEP 3 : VERIFY DATA TYPES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3 : VERIFY DATA TYPES")
print("=" * 70)

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

print(dtype_df)

# =============================================================================
# STEP 4 : CHECK INFINITE VALUES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4 : CHECK INFINITE VALUES")
print("=" * 70)

numeric_columns = df.select_dtypes(include=[np.number]).columns

infinite_df = pd.DataFrame({
    "Infinite Count":
    np.isinf(df[numeric_columns]).sum()
})

print(infinite_df)

# =============================================================================
# STEP 5 : CHECK CONSTANT FEATURES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5 : CONSTANT FEATURES")
print("=" * 70)

constant_features = []

for column in numeric_columns:

    if df[column].nunique() == 1:
        constant_features.append(column)

constant_df = pd.DataFrame({
    "Constant Features": constant_features
})

print(constant_df)

# =============================================================================
# STEP 6 : CHECK NEAR ZERO VARIANCE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6 : NEAR ZERO VARIANCE")
print("=" * 70)

variance = df[numeric_columns].var()

nzv_df = variance.reset_index()

nzv_df.columns = ["Feature", "Variance"]

nzv_df = nzv_df.sort_values("Variance")

print(nzv_df)

# =============================================================================
# STEP 7 : SAVE PREPROCESSED DATASET
# =============================================================================

print("\nSaving preprocessed dataset...")

preprocessed_file = os.path.join(
    CLEAN_DATA_FOLDER,
    "preprocessed_bankruptcy.xlsx"
)

df.to_excel(
    preprocessed_file,
    index=False
)

print("Preprocessed dataset saved successfully.")

# =============================================================================
# STEP 8 : SAVE REPORT
# =============================================================================

print("Saving preprocessing report...")

report_file = os.path.join(
    RESULTS_FOLDER,
    "Preprocessing_Report.xlsx"
)

dataset_summary = pd.DataFrame({
    "Metric": ["Rows", "Columns"],
    "Value": list(df.shape)
})

with pd.ExcelWriter(report_file, engine="openpyxl") as writer:

    dataset_summary.to_excel(
        writer,
        sheet_name="Dataset_Summary",
        index=False
    )

    encoded_target.to_excel(
        writer,
        sheet_name="Encoded_Target",
        index=False
    )

    infinite_df.to_excel(
        writer,
        sheet_name="Infinite_Values"
    )

    constant_df.to_excel(
        writer,
        sheet_name="Constant_Features",
        index=False
    )

    nzv_df.to_excel(
        writer,
        sheet_name="Near_Zero_Variance",
        index=False
    )

print("Preprocessing report saved successfully.")

# =============================================================================
# END
# =============================================================================

print("\n" + "=" * 70)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)