# =============================================================================
# PROJECT : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# SCRIPT  : 08_Exploratory_Data_Analysis.py
# PURPOSE : Exploratory Data Analysis (EDA)
# =============================================================================

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import os
import sys
import warnings

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# ADD PROJECT ROOT TO PYTHON PATH
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
    FIGURES_FOLDER,
    COMPANY_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_step(step, title):
    """Print formatted step heading."""
    print("\n" + "=" * 70)
    print(f"STEP {step} : {title}")
    print("=" * 70)


def create_folder(folder):
    """Create folder if it does not exist."""
    os.makedirs(folder, exist_ok=True)


# =============================================================================
# CREATE OUTPUT FOLDERS
# =============================================================================

create_folder(RESULTS_FOLDER)
create_folder(FIGURES_FOLDER)

# =============================================================================
# STEP 1 : LOAD HYBRID FEATURE DATASET
# =============================================================================

print_step(1, "LOADING HYBRID FEATURE DATASET")

dataset_path = os.path.join(
    FEATURESET_FOLDER,
    "Hybrid_Features.xlsx"
)

df = pd.read_excel(dataset_path)

print("Dataset Loaded Successfully.\n")
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# =============================================================================
# STEP 2 : DATASET SUMMARY
# =============================================================================

print_step(2, "DATASET SUMMARY")

summary_df = pd.DataFrame({
    "Property": [
        "Rows",
        "Columns",
        "Memory Usage (MB)"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    ]
})

print(summary_df)

print("\nColumn Information\n")

column_info = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str)
})

print(column_info)

# =============================================================================
# STEP 3 : TARGET DISTRIBUTION
# =============================================================================

print_step(3, "TARGET DISTRIBUTION")

target_distribution = (
    df[TARGET_COLUMN]
    .value_counts()
    .sort_index()
    .rename_axis("Class")
    .reset_index(name="Count")
)

target_distribution["Class"] = target_distribution["Class"].map({
    0: "Alive",
    1: "Failed"
})

print(target_distribution)

plt.figure(figsize=(6,5))

plt.bar(
    target_distribution["Class"],
    target_distribution["Count"]
)

plt.title("Target Distribution")
plt.xlabel("Company Status")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_FOLDER,
        "Target_Distribution.png"
    ),
    dpi=300
)

plt.close()

print("Target Distribution Figure Saved.")

# =============================================================================
# STEP 4 : YEAR DISTRIBUTION
# =============================================================================

print_step(4, "YEAR DISTRIBUTION")

year_distribution = (
    df[YEAR_COLUMN]
    .value_counts()
    .sort_index()
    .reset_index()
)

year_distribution.columns = [
    "Year",
    "Count"
]

print(year_distribution)

plt.figure(figsize=(10,5))

plt.bar(
    year_distribution["Year"].astype(str),
    year_distribution["Count"]
)

plt.xticks(rotation=90)

plt.title("Year Distribution")

plt.xlabel("Year")

plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_FOLDER,
        "Year_Distribution.png"
    ),
    dpi=300
)

plt.close()

print("Year Distribution Figure Saved.")

# =============================================================================
# STEP 5 : DESCRIPTIVE STATISTICS
# =============================================================================

print_step(5, "DESCRIPTIVE STATISTICS")

exclude_columns = [
    COMPANY_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN
]

numeric_columns = [
    column
    for column in df.columns
    if column not in exclude_columns
]

descriptive_statistics = (
    df[numeric_columns]
    .describe()
    .T
)

print(descriptive_statistics)

print("\nDescriptive Statistics Generated Successfully.")

# =============================================================================
# CREATE EXCEL REPORT
# =============================================================================

excel_file = os.path.join(
    RESULTS_FOLDER,
    "EDA_Report.xlsx"
)

writer = pd.ExcelWriter(
    excel_file,
    engine="openpyxl"
)

summary_df.to_excel(
    writer,
    sheet_name="Dataset Summary",
    index=False
)

column_info.to_excel(
    writer,
    sheet_name="Column Information",
    index=False
)

target_distribution.to_excel(
    writer,
    sheet_name="Target Distribution",
    index=False
)

year_distribution.to_excel(
    writer,
    sheet_name="Year Distribution",
    index=False
)

descriptive_statistics.to_excel(
    writer,
    sheet_name="Descriptive Statistics"
)

# =============================================================================
# STEP 6 : SKEWNESS & KURTOSIS
# =============================================================================

print_step(6, "SKEWNESS & KURTOSIS")

distribution_df = pd.DataFrame({
    "Feature": numeric_columns,
    "Skewness": df[numeric_columns].skew().values,
    "Kurtosis": df[numeric_columns].kurtosis().values
})

print(distribution_df)

distribution_df.to_excel(
    writer,
    sheet_name="Skewness_Kurtosis",
    index=False
)

print("Skewness and Kurtosis Calculated Successfully.")

# =============================================================================
# STEP 7 : HISTOGRAMS
# =============================================================================

print_step(7, "GENERATING HISTOGRAMS")

rows = (len(numeric_columns) + 3) // 4

fig, axes = plt.subplots(
    rows,
    4,
    figsize=(18, rows * 3)
)

axes = axes.flatten()

for i, column in enumerate(numeric_columns):

    axes[i].hist(df[column], bins=30)

    axes[i].set_title(column, fontsize=8)

    axes[i].tick_params(labelsize=6)

# Remove empty subplots
for j in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

histogram_path = os.path.join(
    FIGURES_FOLDER,
    "Histograms.png"
)

plt.savefig(
    histogram_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Histograms Saved Successfully.")

# =============================================================================
# STEP 8 : BOXPLOTS
# =============================================================================

print_step(8, "GENERATING BOXPLOTS")

fig, axes = plt.subplots(
    rows,
    4,
    figsize=(18, rows * 3)
)

axes = axes.flatten()

for i, column in enumerate(numeric_columns):

    axes[i].boxplot(
        df[column].dropna(),
        vert=True
    )

    axes[i].set_title(column, fontsize=8)

    axes[i].tick_params(labelsize=6)

for j in range(len(numeric_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

boxplot_path = os.path.join(
    FIGURES_FOLDER,
    "Boxplots.png"
)

plt.savefig(
    boxplot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Boxplots Saved Successfully.")

# =============================================================================
# STEP 9 : SAVE REPORT
# =============================================================================

print_step(9, "SAVING REPORT")

writer.close()

print(f"EDA Report Saved Successfully : {excel_file}")

print("\nGenerated Figures")

print("---------------------------")

print("1. Target_Distribution.png")

print("2. Year_Distribution.png")

print("3. Histograms.png")

print("4. Boxplots.png")

# =============================================================================
# PROJECT COMPLETED
# =============================================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)