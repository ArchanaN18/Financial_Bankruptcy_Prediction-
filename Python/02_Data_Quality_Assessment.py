# =============================================================================
# Project : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# File    : 02_Data_Quality_Assessment.py
# Phase   : Data Quality Assessment
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
    RESULTS_FOLDER,
    COMPANY_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN
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
print("STEP 1 : LOADING WORKING DATASET")
print("=" * 70)

df = pd.read_excel(WORKING_DATA_FILE)

print("Dataset loaded successfully.")

# =============================================================================
# STEP 2 : DATASET SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2 : DATASET SUMMARY")
print("=" * 70)

rows, columns = df.shape

memory = df.memory_usage(deep=True).sum() / (1024 ** 2)

dataset_summary = pd.DataFrame({

    "Metric": [

        "Number of Rows",
        "Number of Columns",
        "Memory Usage (MB)"

    ],

    "Value": [

        rows,
        columns,
        round(memory, 2)

    ]

})

print(dataset_summary)

# =============================================================================
# STEP 3 : MISSING VALUES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3 : MISSING VALUES")
print("=" * 70)

missing_df = pd.DataFrame({

    "Missing Count": df.isnull().sum(),

    "Missing Percentage": round(df.isnull().mean() * 100, 2)

})

print(missing_df)

# =============================================================================
# STEP 4 : DUPLICATE RECORDS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4 : DUPLICATE RECORDS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

duplicate_df = pd.DataFrame({

    "Metric": ["Duplicate Records"],

    "Value": [duplicate_count]

})

print(duplicate_df)

# =============================================================================
# STEP 5 : COMPANY SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5 : COMPANY SUMMARY")
print("=" * 70)

company_summary = pd.DataFrame({

    "Metric": ["Unique Companies"],

    "Value": [df[COMPANY_COLUMN].nunique()]

})

print(company_summary)

# =============================================================================
# STEP 6 : YEAR SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6 : YEAR SUMMARY")
print("=" * 70)

year_summary = pd.DataFrame(

    sorted(df[YEAR_COLUMN].unique()),

    columns=["Year"]

)

print(year_summary)

# =============================================================================
# STEP 7 : TARGET DISTRIBUTION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7 : TARGET DISTRIBUTION")
print("=" * 70)

target_distribution = df[TARGET_COLUMN].value_counts().reset_index()

target_distribution.columns = ["Status", "Count"]

target_distribution["Percentage"] = round(

    target_distribution["Count"] /

    target_distribution["Count"].sum() * 100,

    2

)

print(target_distribution)

# =============================================================================
# STEP 8 : DATA TYPES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8 : DATA TYPES")
print("=" * 70)

data_types = pd.DataFrame({

    "Column": df.columns,

    "Data Type": df.dtypes.astype(str).values

})

print(data_types)

# =============================================================================
# STEP 9 : SAVE REPORT
# =============================================================================

print("\nSaving Data Quality Report...")

report_file = os.path.join(

    RESULTS_FOLDER,

    "Data_Quality_Report.xlsx"

)

with pd.ExcelWriter(report_file, engine="openpyxl") as writer:

    dataset_summary.to_excel(

        writer,

        sheet_name="Dataset_Summary",

        index=False

    )

    missing_df.to_excel(

        writer,

        sheet_name="Missing_Values"

    )

    duplicate_df.to_excel(

        writer,

        sheet_name="Duplicate_Records",

        index=False

    )

    company_summary.to_excel(

        writer,

        sheet_name="Company_Summary",

        index=False

    )

    year_summary.to_excel(

        writer,

        sheet_name="Year_Summary",

        index=False

    )

    target_distribution.to_excel(

        writer,

        sheet_name="Target_Distribution",

        index=False

    )

    data_types.to_excel(

        writer,

        sheet_name="Data_Types",

        index=False

    )

print("Data Quality Report saved successfully.")

# =============================================================================
# END
# =============================================================================

print("\n" + "=" * 70)
print("DATA QUALITY ASSESSMENT COMPLETED SUCCESSFULLY")
print("=" * 70)