# =============================================================================
# Project : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# File    : 06_Composition_Features.py
# Phase   : Composition Feature Generation
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
    FEATURESET_FOLDER,
    RESULTS_FOLDER
)

# =============================================================================
# Display Settings
# =============================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

# =============================================================================
# STEP 1 : LOAD TRADITIONAL FEATURE DATASET
# =============================================================================

print("=" * 70)
print("STEP 1 : LOADING TRADITIONAL FEATURE DATASET")
print("=" * 70)

input_file = os.path.join(
    FEATURESET_FOLDER,
    "Traditional_Features.xlsx"
)

df = pd.read_excel(input_file)

print("Dataset loaded successfully.")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# =============================================================================
# STEP 2 : CREATE COMPOSITION FEATURES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2 : CREATING COMPOSITION FEATURES")
print("=" * 70)

# ---------------- Asset Structure ----------------

df["Current_Assets_to_Assets"] = df["X1"] / df["X10"]

df["Inventory_to_Assets"] = df["X5"] / df["X10"]

df["Receivables_to_Assets"] = df["X7"] / df["X10"]

df["Retained_Earnings_to_Assets"] = df["X15"] / df["X10"]

# ---------------- Liability Structure ----------------

df["Long_Term_Debt_to_Liabilities"] = df["X11"] / df["X17"]

df["Current_Liabilities_to_Liabilities"] = df["X14"] / df["X17"]

# ---------------- Income Structure ----------------

df["COGS_to_Revenue"] = df["X2"] / df["X16"]

df["Gross_Profit_to_Revenue"] = df["X13"] / df["X16"]

df["Net_Income_to_Revenue"] = df["X6"] / df["X16"]

df["Operating_Expenses_to_Revenue"] = df["X18"] / df["X16"]

print("10 composition features created successfully.")

# =============================================================================
# STEP 3 : HANDLE INFINITE VALUES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3 : HANDLING INFINITE VALUES")
print("=" * 70)

composition_columns = [

    "Current_Assets_to_Assets",
    "Inventory_to_Assets",
    "Receivables_to_Assets",
    "Retained_Earnings_to_Assets",
    "Long_Term_Debt_to_Liabilities",
    "Current_Liabilities_to_Liabilities",
    "COGS_to_Revenue",
    "Gross_Profit_to_Revenue",
    "Net_Income_to_Revenue",
    "Operating_Expenses_to_Revenue"

]

df[composition_columns] = df[composition_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

missing_before = df[composition_columns].isna().sum()

print("Infinite values replaced with NaN.")

# =============================================================================
# STEP 4 : SKIP MEDIAN IMPUTATION (PREVENT DATA LEAKAGE)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4 : SKIP MEDIAN IMPUTATION (PREVENT DATA LEAKAGE)")
print("=" * 70)

print("Median imputation skipped to prevent pre-split data leakage.")
missing_after = df[composition_columns].isna().sum()

# =============================================================================
# STEP 5 : CREATE REPORT
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5 : GENERATING REPORT")
print("=" * 70)

composition_report = pd.DataFrame({

    "Feature Name": composition_columns,

    "Formula": [

        "X1 / X10",
        "X5 / X10",
        "X7 / X10",
        "X15 / X10",
        "X11 / X17",
        "X14 / X17",
        "X2 / X16",
        "X13 / X16",
        "X6 / X16",
        "X18 / X16"

    ],

    "Missing Before": missing_before.values,

    "Missing After": missing_after.values

})

composition_summary = df[composition_columns].describe().T

# =============================================================================
# STEP 6 : SAVE DATASET
# =============================================================================

print("\nSaving Composition Feature Dataset...")

output_file = os.path.join(
    FEATURESET_FOLDER,
    "Composition_Features.xlsx"
)

df.to_excel(
    output_file,
    index=False
)

print("Composition feature dataset saved successfully.")

# =============================================================================
# STEP 7 : SAVE REPORT
# =============================================================================

print("Saving Composition Feature Report...")

report_file = os.path.join(
    RESULTS_FOLDER,
    "Composition_Feature_Report.xlsx"
)

with pd.ExcelWriter(report_file, engine="openpyxl") as writer:

    composition_report.to_excel(
        writer,
        sheet_name="Composition_Details",
        index=False
    )

    composition_summary.to_excel(
        writer,
        sheet_name="Composition_Summary"
    )

print("Composition Feature Report saved successfully.")

# =============================================================================
# END
# =============================================================================

print("\n" + "=" * 70)
print("COMPOSITION FEATURE GENERATION COMPLETED SUCCESSFULLY")
print("=" * 70)