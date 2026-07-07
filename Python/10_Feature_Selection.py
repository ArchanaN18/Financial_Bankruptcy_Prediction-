"""
10_Feature_Selection.py
SKELETON GENERATED

NOTE:
The full implementation requested is too large for a single chat response.
This file is created so it can be extended in-place.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import (
    FEATURESET_FOLDER,
    RESULTS_FOLDER,
    FIGURES_FOLDER,
    COMPANY_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    RANDOM_STATE
)

CORRELATION_THRESHOLD = 0.90

def print_step(step, title):
    print("\n" + "="*70)
    print(f"STEP {step} : {title}")
    print("="*70)

print_step(1, "LOAD DATA")

train_path = os.path.join(FEATURESET_FOLDER, "Train_Company_Split.xlsx")
test_path = os.path.join(FEATURESET_FOLDER, "Test_Company_Split.xlsx")

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

exclude = [COMPANY_COLUMN, TARGET_COLUMN, YEAR_COLUMN]
feature_columns = [c for c in train_df.columns if c not in exclude]

# Temporary imputation for feature selection calculations
X_train = train_df[feature_columns].copy()
for col in feature_columns:
    if X_train[col].isna().any():
        X_train[col] = X_train[col].fillna(X_train[col].median())
    if np.isinf(X_train[col]).any():
        col_median = X_train[col].median()
        X_train[col] = X_train[col].replace([np.inf, -np.inf], col_median)
y_train = train_df[TARGET_COLUMN]

print_step(2, "CORRELATION")

corr = X_train.corr(method="pearson")

print_step(3, "MUTUAL INFORMATION")

mi = mutual_info_classif(
    X_train,
    y_train,
    random_state=RANDOM_STATE
)

mi_scores = pd.Series(mi, index=feature_columns).sort_values(ascending=False)

print(mi_scores.head())

# =============================================================================
# STEP 4 : IDENTIFY HIGHLY CORRELATED FEATURE PAIRS
# =============================================================================

print_step(4, "IDENTIFYING HIGHLY CORRELATED FEATURES")

correlation_pairs = []

columns = corr.columns.tolist()

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        feature_1 = columns[i]
        feature_2 = columns[j]

        correlation = corr.loc[feature_1, feature_2]

        if abs(correlation) >= CORRELATION_THRESHOLD:

            correlation_pairs.append({

                "Feature_1": feature_1,
                "Feature_2": feature_2,
                "Correlation": correlation,
                "MI_1": mi_scores[feature_1],
                "MI_2": mi_scores[feature_2]

            })

correlation_pairs_df = pd.DataFrame(correlation_pairs)

print(f"Highly Correlated Pairs : {len(correlation_pairs_df)}")

# =============================================================================
# STEP 5 : REMOVE REDUNDANT FEATURES USING MUTUAL INFORMATION
# =============================================================================

print_step(5, "REMOVING REDUNDANT FEATURES")

# Sort pairs by strongest correlation first
correlation_pairs_df["Abs_Correlation"] = (
    correlation_pairs_df["Correlation"].abs()
)

correlation_pairs_df = correlation_pairs_df.sort_values(
    by="Abs_Correlation",
    ascending=False
).reset_index(drop=True)

removed_features = set()
protected_features = set()
kept_features = []

for _, row in correlation_pairs_df.iterrows():

    feature_1 = row["Feature_1"]
    feature_2 = row["Feature_2"]

    # Ignore if already removed
    if feature_1 in removed_features or feature_2 in removed_features:
        continue

    # If both already protected, ignore
    if feature_1 in protected_features and feature_2 in protected_features:
        continue

    # If one feature has already been protected,
    # remove the other immediately
    if feature_1 in protected_features:

        kept = feature_1
        removed = feature_2

    elif feature_2 in protected_features:

        kept = feature_2
        removed = feature_1

    else:

        if row["MI_1"] >= row["MI_2"]:

            kept = feature_1
            removed = feature_2

        else:

            kept = feature_2
            removed = feature_1

    protected_features.add(kept)
    removed_features.add(removed)

    kept_features.append({

        "Kept_Feature": kept,
        "Removed_Feature": removed,
        "Correlation": row["Correlation"],
        "MI_Kept": max(row["MI_1"], row["MI_2"]),
        "MI_Removed": min(row["MI_1"], row["MI_2"])

    })

removed_features = sorted(list(removed_features))

selected_features = [

    feature

    for feature in feature_columns

    if feature not in removed_features

]

print(f"Original Features : {len(feature_columns)}")
print(f"Removed Features  : {len(removed_features)}")
print(f"Selected Features : {len(selected_features)}")


# =============================================================================
# STEP 6 : VERIFY FEATURE SELECTION AND CREATE SELECTED DATASETS
# =============================================================================

print_step(6, "VERIFY FEATURE SELECTION")

print("\n" + "=" * 70)
print("REMOVED FEATURES")
print("=" * 70)

for feature in removed_features:
    print(feature)

kept_removed_df = pd.DataFrame(kept_features)

print("\n" + "=" * 70)
print("FEATURE REMOVAL DECISIONS")
print("=" * 70)

print(kept_removed_df)

print("\n" + "=" * 70)
print("CREATING SELECTED DATASETS")
print("=" * 70)

# Columns to retain
selected_columns = (
    [COMPANY_COLUMN, YEAR_COLUMN]
    + selected_features
    + [TARGET_COLUMN]
)

# Create training dataset
train_selected = train_df[selected_columns].copy()

# Create testing dataset
test_selected = test_df[selected_columns].copy()

print(f"Training Dataset Shape : {train_selected.shape}")
print(f"Testing Dataset Shape  : {test_selected.shape}")

print(f"\nSelected Features ({len(selected_features)}):")

for i, feature in enumerate(selected_features, start=1):
    print(f"{i:2d}. {feature}")

# =============================================================================
# STEP 7 : CORRELATION HEATMAP AFTER FEATURE SELECTION
# =============================================================================

print_step(7, "CORRELATION HEATMAP AFTER FEATURE SELECTION")

corr_selected = train_selected[selected_features].corr(method="pearson")

plt.figure(figsize=(12, 10))

plt.imshow(corr_selected, aspect="auto")

plt.colorbar()

plt.xticks(
    range(len(selected_features)),
    selected_features,
    rotation=90,
    fontsize=6
)

plt.yticks(
    range(len(selected_features)),
    selected_features,
    fontsize=6
)

plt.title("Correlation Heatmap (After Feature Selection)")

plt.tight_layout()

heatmap_after_path = os.path.join(
    FIGURES_FOLDER,
    "Correlation_Heatmap_After.png"
)

plt.savefig(
    heatmap_after_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Correlation Heatmap Saved.")

# =============================================================================
# STEP 8 : MUTUAL INFORMATION PLOT
# =============================================================================

print_step(8, "MUTUAL INFORMATION PLOT")

mi_final = mi_scores[selected_features].sort_values(ascending=False)

plt.figure(figsize=(10, 8))

plt.barh(
    mi_final.index,
    mi_final.values
)

plt.gca().invert_yaxis()

plt.xlabel("Mutual Information")

plt.title("Mutual Information Scores")

plt.tight_layout()

mi_plot_path = os.path.join(
    FIGURES_FOLDER,
    "Mutual_Information.png"
)

plt.savefig(
    mi_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Mutual Information Plot Saved.")

# =============================================================================
# STEP 9 : SAVE SELECTED DATASETS
# =============================================================================

print_step(9, "SAVING SELECTED DATASETS")

train_output = os.path.join(
    FEATURESET_FOLDER,
    "Train_Selected_Features.xlsx"
)

test_output = os.path.join(
    FEATURESET_FOLDER,
    "Test_Selected_Features.xlsx"
)

train_selected.to_excel(
    train_output,
    index=False
)

test_selected.to_excel(
    test_output,
    index=False
)

print("Training Dataset Saved.")
print("Testing Dataset Saved.")

# =============================================================================
# STEP 10 : SAVE FEATURE SELECTION REPORT
# =============================================================================

print_step(10, "SAVING FEATURE SELECTION REPORT")

summary_df = pd.DataFrame({

    "Metric": [

        "Original Features",

        "Removed Features",

        "Selected Features"

    ],

    "Value": [

        len(feature_columns),

        len(removed_features),

        len(selected_features)

    ]

})

removed_df = pd.DataFrame({

    "Removed_Feature": removed_features

})

selected_df = pd.DataFrame({

    "Selected_Feature": selected_features

})

mi_report = (
    mi_scores
    .sort_values(ascending=False)
    .rename("Mutual_Information")
    .reset_index()
)

mi_report.columns = [

    "Feature",

    "Mutual_Information"

]

report_path = os.path.join(
    RESULTS_FOLDER,
    "Feature_Selection_Report.xlsx"
)

with pd.ExcelWriter(
    report_path,
    engine="openpyxl"
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    correlation_pairs_df.to_excel(
        writer,
        sheet_name="Correlated_Pairs",
        index=False
    )

    pd.DataFrame(corr).to_excel(
        writer,
        sheet_name="Correlation_Matrix"
    )

    mi_report.to_excel(
        writer,
        sheet_name="Mutual_Information",
        index=False
    )

    removed_df.to_excel(
        writer,
        sheet_name="Removed_Features",
        index=False
    )

    selected_df.to_excel(
        writer,
        sheet_name="Selected_Features",
        index=False
    )

print("Feature Selection Report Saved.")

# =============================================================================
# PROJECT COMPLETED
# =============================================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)