# =============================================================================
# PROJECT : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# SCRIPT  : 09_Company_Based_Split.py
# PURPOSE : Company-Based Train/Test Split
# =============================================================================

import os
import sys
import warnings
import pandas as pd

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# ADD PROJECT ROOT
# =============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# =============================================================================
# IMPORT CONFIGURATION
# =============================================================================

from config import (
    FEATURESET_FOLDER,
    RESULTS_FOLDER,
    COMPANY_COLUMN,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)

# =============================================================================
# HELPER FUNCTION
# =============================================================================

def print_step(step, title):
    print("\n" + "=" * 70)
    print(f"STEP {step} : {title}")
    print("=" * 70)

# =============================================================================
# STEP 1 : LOAD HYBRID DATASET
# =============================================================================

print_step(1, "LOADING HYBRID DATASET")

dataset_path = os.path.join(
    FEATURESET_FOLDER,
    "Hybrid_Features.xlsx"
)

df = pd.read_excel(dataset_path)

print("Dataset Loaded Successfully.")
print(f"Rows : {len(df):,}")

# =============================================================================
# STEP 2 : EXTRACT UNIQUE COMPANIES
# =============================================================================

print_step(2, "EXTRACTING UNIQUE COMPANIES")

companies = sorted(df[COMPANY_COLUMN].unique())

print(f"Total Companies : {len(companies):,}")

# =============================================================================
# STEP 3 : COMPANY-BASED SPLIT
# =============================================================================

print_step(3, "PERFORMING COMPANY SPLIT")

train_companies, test_companies = train_test_split(
    companies,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

print(f"Training Companies : {len(train_companies):,}")
print(f"Testing Companies  : {len(test_companies):,}")

# =============================================================================
# STEP 4 : CREATE TRAIN AND TEST DATASETS
# =============================================================================

print_step(4, "CREATING TRAINING AND TESTING DATASETS")

train_df = df[
    df[COMPANY_COLUMN].isin(train_companies)
].copy()

test_df = df[
    df[COMPANY_COLUMN].isin(test_companies)
].copy()

print(f"Training Records : {len(train_df):,}")
print(f"Testing Records  : {len(test_df):,}")

# =============================================================================
# STEP 5 : VALIDATE COMPANY OVERLAP
# =============================================================================

print_step(5, "VALIDATING COMPANY OVERLAP")

overlap = (
    set(train_df[COMPANY_COLUMN])
    &
    set(test_df[COMPANY_COLUMN])
)

if overlap:
    raise ValueError(
        f"{len(overlap)} companies found in both datasets."
    )

print("No Company Overlap Detected.")

# =============================================================================
# STEP 6 : TARGET DISTRIBUTION
# =============================================================================

print_step(6, "TARGET DISTRIBUTION")

train_target = (
    train_df[TARGET_COLUMN]
    .value_counts()
    .rename("Train")
)

test_target = (
    test_df[TARGET_COLUMN]
    .value_counts()
    .rename("Test")
)

target_report = pd.concat(
    [train_target, test_target],
    axis=1
).fillna(0).astype(int)

print(target_report)

# =============================================================================
# STEP 7 : SAVE DATASETS
# =============================================================================

print_step(7, "SAVING DATASETS")

train_path = os.path.join(
    FEATURESET_FOLDER,
    "Train_Company_Split.xlsx"
)

test_path = os.path.join(
    FEATURESET_FOLDER,
    "Test_Company_Split.xlsx"
)

train_df.to_excel(
    train_path,
    index=False
)

test_df.to_excel(
    test_path,
    index=False
)

print("Training Dataset Saved.")
print("Testing Dataset Saved.")

# =============================================================================
# STEP 8 : SAVE REPORT
# =============================================================================

print_step(8, "SAVING SPLIT REPORT")

summary = pd.DataFrame({
    "Metric": [
        "Total Companies",
        "Training Companies",
        "Testing Companies",
        "Training Records",
        "Testing Records"
    ],
    "Value": [
        len(companies),
        len(train_companies),
        len(test_companies),
        len(train_df),
        len(test_df)
    ]
})

report_path = os.path.join(
    RESULTS_FOLDER,
    "Company_Split_Report.xlsx"
)

with pd.ExcelWriter(
    report_path,
    engine="openpyxl"
) as writer:

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    target_report.to_excel(
        writer,
        sheet_name="Target Distribution"
    )

print("Company Split Report Saved.")

# =============================================================================
# PROJECT COMPLETED
# =============================================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)