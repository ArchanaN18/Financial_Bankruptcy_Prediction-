# =============================================================================
# PROJECT : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# SCRIPT  : 11_SMOTE.py
# PURPOSE : Apply SMOTE on Training Dataset
# =============================================================================

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import os
import sys
import warnings

import pandas as pd



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
    YEAR_COLUMN,
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
# CREATE OUTPUT FOLDER
# =============================================================================

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)

# =============================================================================
# STEP 1 : LOAD TRAINING DATASET
# =============================================================================

print_step(1, "LOADING TRAINING DATASET")

train_path = os.path.join(
    FEATURESET_FOLDER,
    "Train_Selected_Features.xlsx"
)

train_df = pd.read_excel(train_path)

print("Training Dataset Loaded Successfully.\n")

print(f"Rows    : {len(train_df):,}")
print(f"Columns : {len(train_df.columns)}")

# =============================================================================
# STEP 2 : SEPARATE FEATURES AND TARGET
# =============================================================================

print_step(2, "SEPARATING FEATURES AND TARGET")

exclude_columns = [
    COMPANY_COLUMN,
    YEAR_COLUMN,
    TARGET_COLUMN
]

feature_columns = [

    column

    for column in train_df.columns

    if column not in exclude_columns

]

X_train = train_df[feature_columns]

y_train = train_df[TARGET_COLUMN]

print(f"Selected Features : {len(feature_columns)}")

print(f"Target Column     : {TARGET_COLUMN}")

# =============================================================================
# STEP 3 : CLASS DISTRIBUTION BEFORE SMOTE
# =============================================================================

print_step(3, "CLASS DISTRIBUTION BEFORE SMOTE")

before_smote = (
    y_train
    .value_counts()
    .sort_index()
)

before_report = pd.DataFrame({

    "Class": before_smote.index,

    "Count": before_smote.values

})

print(before_report)

# =============================================================================
# STEP 4 : APPLY SMOTE
# =============================================================================

print_step(4, "APPLYING SMOTE")
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek

print("Handling infinite values and imputing missing data before SMOTE...")
X_train = X_train.replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)

smote = SMOTETomek(
    random_state=RANDOM_STATE
)

X_resampled, y_resampled = smote.fit_resample(
    X_train_imputed,
    y_train
)

print("SMOTE Applied Successfully.")

print(f"Rows Before SMOTE : {len(X_train):,}")

print(f"Rows After SMOTE  : {len(X_resampled):,}")

# =============================================================================
# STEP 5 : CLASS DISTRIBUTION AFTER SMOTE
# =============================================================================

print_step(5, "CLASS DISTRIBUTION AFTER SMOTE")

after_smote = (
    pd.Series(y_resampled)
    .value_counts()
    .sort_index()
)

after_report = pd.DataFrame({

    "Class": after_smote.index,

    "Count": after_smote.values

})

print(after_report)

# =============================================================================
# STEP 6 : CREATE BALANCED DATASET
# =============================================================================

print_step(6, "CREATING BALANCED DATASET")

train_smote = pd.DataFrame(
    X_resampled,
    columns=feature_columns
)

train_smote[TARGET_COLUMN] = y_resampled

print(f"Balanced Dataset Shape : {train_smote.shape}")

# =============================================================================
# STEP 7 : SAVE BALANCED DATASET
# =============================================================================

print_step(7, "SAVING BALANCED DATASET")

smote_dataset_path = os.path.join(
    FEATURESET_FOLDER,
    "Train_SMOTE.xlsx"
)

train_smote.to_excel(
    smote_dataset_path,
    index=False
)

print("Balanced Training Dataset Saved.")

# =============================================================================
# STEP 8 : SAVE SMOTE REPORT
# =============================================================================

print_step(8, "SAVING SMOTE REPORT")

summary = pd.DataFrame({

    "Metric":[

        "Rows Before SMOTE",

        "Rows After SMOTE",

        "Selected Features"

    ],

    "Value":[

        len(train_df),

        len(train_smote),

        len(feature_columns)

    ]

})

report_path = os.path.join(
    RESULTS_FOLDER,
    "SMOTE_Report.xlsx"
)

with pd.ExcelWriter(
    report_path,
    engine="openpyxl"
) as writer:

    before_report.to_excel(
        writer,
        sheet_name="Before_SMOTE",
        index=False
    )

    after_report.to_excel(
        writer,
        sheet_name="After_SMOTE",
        index=False
    )

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

print("SMOTE Report Saved.")

# =============================================================================
# PROJECT COMPLETED
# =============================================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)