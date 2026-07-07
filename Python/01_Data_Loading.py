# =============================================================================
# Project : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# File    : 01_Data_Loading.py
# Phase   : Dataset Understanding
# =============================================================================
import os
import sys
import pandas as pd


# =============================================================================
# Allow Python to locate config.py
# =============================================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

# =============================================================================
# Import Project Configuration
# =============================================================================

from config import (
    RAW_DATA_FILE,
    WORKING_DATA_FILE,
    RESULTS_FOLDER
)

# =============================================================================
# Display Settings
# =============================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# =============================================================================
# Step 1 : Load Dataset
# =============================================================================

print("=" * 70)
print("STEP 1 : LOADING DATASET")
print("=" * 70)

df = pd.read_excel(RAW_DATA_FILE)

print("Dataset loaded successfully.")

# =============================================================================
# Step 2 : Create Working Copy
# =============================================================================

working_df = df.copy()

print("Working copy created.")

# =============================================================================
# Step 3 : Dataset Shape
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2 : DATASET SHAPE")
print("=" * 70)

rows, columns = working_df.shape

print(f"Rows    : {rows}")
print(f"Columns : {columns}")

# =============================================================================
# Step 4 : Column Names
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3 : COLUMN NAMES")
print("=" * 70)

column_df = pd.DataFrame(
    working_df.columns,
    columns=["Column Name"]
)

print(column_df)

# =============================================================================
# Step 5 : Data Types
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4 : DATA TYPES")
print("=" * 70)

dtype_df = pd.DataFrame(
    working_df.dtypes,
    columns=["Data Type"]
)

print(dtype_df)

# =============================================================================
# Step 6 : First Five Records
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5 : FIRST FIVE RECORDS")
print("=" * 70)

print(working_df.head())

# =============================================================================
# Step 7 : Last Five Records
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6 : LAST FIVE RECORDS")
print("=" * 70)

print(working_df.tail())

# =============================================================================
# Step 8 : Statistical Summary
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7 : DESCRIPTIVE STATISTICS")
print("=" * 70)

summary_df = working_df.describe(include="all")

print(summary_df)

# =============================================================================
# Step 9 : Memory Usage
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8 : MEMORY USAGE")
print("=" * 70)

memory = working_df.memory_usage(deep=True).sum() / (1024**2)

print(f"Memory Usage : {memory:.2f} MB")

# =============================================================================
# Step 10 : Save Results
# =============================================================================

# =============================================================================
# Step 10 : Save Results
# =============================================================================

print("\nSaving Results...")

profile_file = os.path.join(
    RESULTS_FOLDER,
    "Dataset_Profile.xlsx"
)

with pd.ExcelWriter(profile_file, engine="openpyxl") as writer:

    # Column Names
    column_df.to_excel(
        writer,
        sheet_name="Column_Names",
        index=False
    )

    # Data Types
    dtype_df.to_excel(
        writer,
        sheet_name="Data_Types"
    )

    # Descriptive Statistics
    summary_df.to_excel(
        writer,
        sheet_name="Descriptive_Statistics"
    )

# Save Working Dataset
working_df.to_excel(
    WORKING_DATA_FILE,
    index=False
)

print("Results saved successfully.")

# =============================================================================
# End
# =============================================================================

print("\n" + "=" * 70)
print("DATA LOADING COMPLETED SUCCESSFULLY")
print("=" * 70)