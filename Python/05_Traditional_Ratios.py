# =============================================================================
# Project : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# File    : 05_Traditional_Ratios.py
# Phase   : Traditional Financial Ratio Generation
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
    CLEAN_DATA_FOLDER,
    FEATURESET_FOLDER,
    RESULTS_FOLDER
)

# =============================================================================
# Display Settings
# =============================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

# =============================================================================
# STEP 1 : LOAD PREPROCESSED DATASET
# =============================================================================

print("=" * 70)
print("STEP 1 : LOADING PREPROCESSED DATASET")
print("=" * 70)

input_file = os.path.join(
    CLEAN_DATA_FOLDER,
    "preprocessed_bankruptcy.xlsx"
)

df = pd.read_excel(input_file)

print("Dataset loaded successfully.")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# =============================================================================
# STEP 2 : CREATE TRADITIONAL FINANCIAL RATIOS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2 : CREATING TRADITIONAL FINANCIAL RATIOS")
print("=" * 70)

# -------------------------------------------------------------------------
# Liquidity Ratios
# -------------------------------------------------------------------------

df["Current_Ratio"] = df["X1"] / df["X14"]

df["Working_Capital_Ratio"] = (
    (df["X1"] - df["X14"]) / df["X10"]
)

# -------------------------------------------------------------------------
# Profitability Ratio
# -------------------------------------------------------------------------

df["ROA"] = df["X6"] / df["X10"]

# -------------------------------------------------------------------------
# Efficiency Ratios
# -------------------------------------------------------------------------

df["Asset_Turnover"] = df["X16"] / df["X10"]

df["Inventory_Turnover"] = df["X2"] / df["X5"]

df["Receivables_Turnover"] = df["X16"] / df["X7"]

# -------------------------------------------------------------------------
# Leverage Ratios
# -------------------------------------------------------------------------

df["Debt_Ratio"] = df["X17"] / df["X10"]

df["Long_Term_Debt_Ratio"] = df["X11"] / df["X10"]

print("8 traditional financial ratios created successfully.")

# =============================================================================
# STEP 3 : HANDLE INFINITE VALUES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3 : HANDLING INFINITE VALUES")
print("=" * 70)

ratio_columns = [

    "Current_Ratio",
    "Working_Capital_Ratio",
    "ROA",
    "Asset_Turnover",
    "Inventory_Turnover",
    "Receivables_Turnover",
    "Debt_Ratio",
    "Long_Term_Debt_Ratio"

]

# Replace Infinity with NaN

df[ratio_columns] = df[ratio_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

missing_before = df[ratio_columns].isna().sum()

print("Infinite values replaced with NaN.")

# =============================================================================
# STEP 4 : SKIP MEDIAN IMPUTATION (PREVENT DATA LEAKAGE)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4 : SKIP MEDIAN IMPUTATION (PREVENT DATA LEAKAGE)")
print("=" * 70)

print("Median imputation skipped to prevent pre-split data leakage.")
missing_after = df[ratio_columns].isna().sum()

# =============================================================================
# STEP 5 : CREATE REPORT TABLES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5 : GENERATING REPORT")
print("=" * 70)

ratio_report = pd.DataFrame({

    "Ratio Name": ratio_columns,

    "Formula": [

        "X1 / X14",
        "(X1 - X14) / X10",
        "X6 / X10",
        "X16 / X10",
        "X2 / X5",
        "X16 / X7",
        "X17 / X10",
        "X11 / X10"

    ],

    "Missing Before": missing_before.values,

    "Missing After": missing_after.values

})

ratio_summary = df[ratio_columns].describe().T

# =============================================================================
# STEP 6 : SAVE FEATURE DATASET
# =============================================================================

print("\nSaving Traditional Feature Dataset...")

feature_file = os.path.join(
    FEATURESET_FOLDER,
    "Traditional_Features.xlsx"
)

df.to_excel(
    feature_file,
    index=False
)

print("Traditional feature dataset saved successfully.")

# =============================================================================
# STEP 7 : SAVE REPORT
# =============================================================================

print("Saving Traditional Ratio Report...")

report_file = os.path.join(
    RESULTS_FOLDER,
    "Traditional_Ratio_Report.xlsx"
)

with pd.ExcelWriter(report_file, engine="openpyxl") as writer:

    ratio_report.to_excel(
        writer,
        sheet_name="Ratio_Details",
        index=False
    )

    ratio_summary.to_excel(
        writer,
        sheet_name="Ratio_Summary"
    )

print("Traditional Ratio Report saved successfully.")

# =============================================================================
# END
# =============================================================================

print("\n" + "=" * 70)
print("TRADITIONAL RATIO GENERATION COMPLETED SUCCESSFULLY")
print("=" * 70)