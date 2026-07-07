# =============================================================================
# Project : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# File    : 07_Hybrid_Features.py
# Phase   : Hybrid Feature Validation & Finalization
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
# STEP 1 : LOAD COMPOSITION FEATURE DATASET
# =============================================================================

print("=" * 70)
print("STEP 1 : LOADING COMPOSITION FEATURE DATASET")
print("=" * 70)

input_file = os.path.join(
    FEATURESET_FOLDER,
    "Composition_Features.xlsx"
)

df = pd.read_excel(input_file)

print("Dataset loaded successfully.")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# =============================================================================
# STEP 2 : DEFINE FEATURE GROUPS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2 : IDENTIFY FEATURE GROUPS")
print("=" * 70)

original_features = [

    "X1","X2","X3","X4","X5","X6","X7","X8","X9",
    "X10","X11","X12","X13","X14","X15","X16","X17","X18"

]

traditional_ratios = [

    "Current_Ratio",
    "Working_Capital_Ratio",
    "ROA",
    "Asset_Turnover",
    "Inventory_Turnover",
    "Receivables_Turnover",
    "Debt_Ratio",
    "Long_Term_Debt_Ratio"

]

composition_features = [

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

# =============================================================================
# STEP 3 : VERIFY FEATURE EXISTENCE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3 : VERIFY FEATURES")
print("=" * 70)

missing_features = []

for feature in (
    original_features +
    traditional_ratios +
    composition_features
):

    if feature not in df.columns:
        missing_features.append(feature)

if len(missing_features) == 0:

    print("All expected features are present.")

else:

    print("Missing Features:")

    for item in missing_features:
        print(item)

# =============================================================================
# STEP 4 : DATASET VALIDATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4 : DATASET VALIDATION")
print("=" * 70)

duplicate_columns = df.columns[df.columns.duplicated()].tolist()

missing_values = df.isnull().sum().sum()

infinite_values = np.isinf(
    df.select_dtypes(include=np.number)
).sum().sum()

validation_summary = pd.DataFrame({

    "Metric":[

        "Rows",
        "Columns",
        "Original Variables",
        "Traditional Ratios",
        "Composition Features",
        "Total Missing Values",
        "Total Infinite Values",
        "Duplicate Columns"

    ],

    "Value":[

        df.shape[0],
        df.shape[1],
        len(original_features),
        len(traditional_ratios),
        len(composition_features),
        missing_values,
        infinite_values,
        len(duplicate_columns)

    ]

})

print(validation_summary)

# =============================================================================
# STEP 5 : FEATURE INVENTORY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5 : FEATURE INVENTORY")
print("=" * 70)

feature_inventory = pd.DataFrame({

    "Feature Group":[

        "Original Variables",
        "Traditional Ratios",
        "Composition Features"

    ],

    "Number of Features":[

        len(original_features),
        len(traditional_ratios),
        len(composition_features)

    ]

})

print(feature_inventory)

# =============================================================================
# STEP 6 : SAVE HYBRID DATASET
# =============================================================================

print("\nSaving Hybrid Feature Dataset...")

output_file = os.path.join(

    FEATURESET_FOLDER,
    "Hybrid_Features.xlsx"

)

df.to_excel(

    output_file,
    index=False

)

print("Hybrid Feature Dataset saved successfully.")

# =============================================================================
# STEP 7 : SAVE REPORT
# =============================================================================

print("Saving Hybrid Feature Report...")

report_file = os.path.join(

    RESULTS_FOLDER,
    "Hybrid_Feature_Report.xlsx"

)

with pd.ExcelWriter(report_file, engine="openpyxl") as writer:

    validation_summary.to_excel(

        writer,
        sheet_name="Validation_Summary",
        index=False

    )

    feature_inventory.to_excel(

        writer,
        sheet_name="Feature_Inventory",
        index=False

    )

print("Hybrid Feature Report saved successfully.")

# =============================================================================
# END
# =============================================================================

print("\n" + "=" * 70)
print("HYBRID FEATURE VALIDATION COMPLETED SUCCESSFULLY")
print("=" * 70)