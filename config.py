# =============================================================================
# CONFIGURATION FILE
# =============================================================================
# Project:
# Hybrid Financial Representation Stacking Ensemble (HFRSE)
# for Explainable Corporate Bankruptcy Prediction
#
# Purpose:
# This file stores all common project settings such as file paths,
# column names, target mapping, random seed, and evaluation settings.
# =============================================================================

import os

# =============================================================================
# PROJECT ROOT DIRECTORY
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# PROJECT FOLDERS
# =============================================================================

DATA_FOLDER = os.path.join(PROJECT_ROOT, "Data")

RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, "Raw")

CLEAN_DATA_FOLDER = os.path.join(DATA_FOLDER, "Clean")

FEATURESET_FOLDER = os.path.join(DATA_FOLDER, "FeatureSets")

PYTHON_FOLDER = os.path.join(PROJECT_ROOT, "Python")

RESULTS_FOLDER = os.path.join(PROJECT_ROOT, "Results")

FIGURES_FOLDER = os.path.join(PROJECT_ROOT, "Figures")

MODELS_FOLDER = os.path.join(PROJECT_ROOT, "Models")

THESIS_FOLDER = os.path.join(PROJECT_ROOT, "Thesis")

PAPER_FOLDER = os.path.join(PROJECT_ROOT, "Paper")

# =============================================================================
# DATASET FILES
# =============================================================================

RAW_DATA_FILE = os.path.join(
    RAW_DATA_FOLDER,
    "bankruptcy.xlsx"
)

WORKING_DATA_FILE = os.path.join(
    CLEAN_DATA_FOLDER,
    "working_bankruptcy.xlsx"
)

# =============================================================================
# COLUMN NAMES
# =============================================================================

COMPANY_COLUMN = "company_name"

TARGET_COLUMN = "status_label"

YEAR_COLUMN = "year"

# =============================================================================
# TARGET LABELS
# =============================================================================

TARGET_MAPPING = {
    "alive": 0,
    "failed": 1
}

# =============================================================================
# RANDOM SEED
# =============================================================================

RANDOM_STATE = 42

# =============================================================================
# TRAIN TEST SPLIT
# =============================================================================

TEST_SIZE = 0.20

# =============================================================================
# CROSS VALIDATION
# =============================================================================

CV_FOLDS = 5

CV_REPEATS = 3

# =============================================================================
# OPTUNA SETTINGS
# =============================================================================

OPTUNA_TRIALS = 50

# =============================================================================
# SHAP SETTINGS
# =============================================================================

SHAP_SAMPLE_SIZE = 5000

# =============================================================================
# AUTOMATICALLY CREATE PROJECT FOLDERS
# =============================================================================

PROJECT_FOLDERS = [

    DATA_FOLDER,

    RAW_DATA_FOLDER,

    CLEAN_DATA_FOLDER,

    FEATURESET_FOLDER,

    RESULTS_FOLDER,

    FIGURES_FOLDER,

    MODELS_FOLDER,

    THESIS_FOLDER,

    PAPER_FOLDER

]

for folder in PROJECT_FOLDERS:

    os.makedirs(folder, exist_ok=True)

# =============================================================================
# END OF CONFIGURATION FILE
# =============================================================================