# =============================================================================
# Project : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# File    : 03_Data_Dictionary.py
# Phase   : Data Dictionary
# =============================================================================

import os
import sys
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
    RESULTS_FOLDER
)

from variable_mapping import VARIABLES

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
# STEP 2 : BUILD DATA DICTIONARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2 : BUILDING DATA DICTIONARY")
print("=" * 70)

records = []

for column in df.columns:

    if column in VARIABLES:

        records.append({

            "Variable": column,

            "Financial Name": VARIABLES[column]["name"],

            "Description": VARIABLES[column]["description"],

            "Financial Statement": VARIABLES[column]["statement"],

            "Category": VARIABLES[column]["category"],

            "Data Type": str(df[column].dtype),

            "Unit": VARIABLES[column]["unit"],

            "Feature Group": VARIABLES[column]["feature_group"]

        })

    else:

        records.append({

            "Variable": column,

            "Financial Name": column,

            "Description": "Dataset Identifier",

            "Financial Statement": "-",

            "Category": "Identifier",

            "Data Type": str(df[column].dtype),

            "Unit": "-",

            "Feature Group": "Metadata"

        })

data_dictionary = pd.DataFrame(records)

print(data_dictionary)

# =============================================================================
# STEP 3 : SAVE REPORT
# =============================================================================

print("\nSaving Data Dictionary...")

report_file = os.path.join(

    RESULTS_FOLDER,

    "Data_Dictionary.xlsx"

)

with pd.ExcelWriter(report_file, engine="openpyxl") as writer:

    data_dictionary.to_excel(

        writer,

        sheet_name="Variable_Dictionary",

        index=False

    )

print("Data Dictionary saved successfully.")

# =============================================================================
# END
# =============================================================================

print("\n" + "=" * 70)
print("DATA DICTIONARY COMPLETED SUCCESSFULLY")
print("=" * 70)