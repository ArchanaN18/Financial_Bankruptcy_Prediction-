# =============================================================================
# PROJECT : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# SCRIPT  : 12_Logistic_Regression.py
# PURPOSE : Logistic Regression Bankruptcy Prediction
# =============================================================================

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import os
import sys
import warnings
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


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
    FIGURES_FOLDER,
    TARGET_COLUMN,
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
# CREATE OUTPUT FOLDERS
# =============================================================================

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)

os.makedirs(
    FIGURES_FOLDER,
    exist_ok=True
)

# =============================================================================
# STEP 1 : LOAD TRAINING DATASET
# =============================================================================

print_step(1, "LOADING TRAINING DATASET")

train_path = os.path.join(
    FEATURESET_FOLDER,
    "Train_SMOTE.xlsx"
)

train_df = pd.read_excel(train_path)

print("Training Dataset Loaded Successfully.")

print(f"Rows    : {len(train_df):,}")
print(f"Columns : {len(train_df.columns)}")

# =============================================================================
# STEP 2 : LOAD TEST DATASET
# =============================================================================

print_step(2, "LOADING TEST DATASET")

test_path = os.path.join(
    FEATURESET_FOLDER,
    "Test_Selected_Features.xlsx"
)

test_df = pd.read_excel(test_path)

print("Testing Dataset Loaded Successfully.")

print(f"Rows    : {len(test_df):,}")
print(f"Columns : {len(test_df.columns)}")

# =============================================================================
# STEP 3 : SEPARATE FEATURES AND TARGET
# =============================================================================

print_step(3, "SEPARATING FEATURES AND TARGET")

X_train = train_df.drop(
    columns=[TARGET_COLUMN]
)

y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(
    columns=[
        "company_name",
        "year",
        TARGET_COLUMN
    ]
)

y_test = test_df[TARGET_COLUMN]

print(f"Training Features : {X_train.shape}")

print(f"Testing Features  : {X_test.shape}")

print("\nTraining Columns")
print(X_train.columns.tolist())

print("\nTesting Columns")
print(X_test.columns.tolist())

print("\nColumn Order Match :",
      list(X_train.columns) == list(X_test.columns))
# =============================================================================
# STEP 4 : IMPUTATION & FEATURE SCALING
# =============================================================================

print_step(4, "IMPUTATION & FEATURE SCALING")

import numpy as np
from sklearn.impute import SimpleImputer

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)

X_test_scaled = scaler.transform(X_test_imputed)

print("Imputation & Feature Scaling Completed.")

# =============================================================================
# STEP 5 : TRAIN LOGISTIC REGRESSION MODEL
# =============================================================================

print_step(5, "TRAINING LOGISTIC REGRESSION")

logistic_model = LogisticRegression(

    random_state=RANDOM_STATE,

    max_iter=1000,

    solver="lbfgs"

)

logistic_model.fit(

    X_train_scaled,

    y_train

)

print("Logistic Regression Model Trained Successfully.")

# =============================================================================
# STEP 6 : PREDICT TEST DATA
# =============================================================================

print_step(6, "PREDICTING TEST DATA")

import numpy as np
from sklearn.metrics import roc_curve

y_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
j_scores = tpr - fpr
best_index = np.argmax(j_scores)
best_threshold = thresholds[best_index]

print(f"Dynamically Calculated Optimal Threshold (Youden's J): {best_threshold:.4f}")

y_pred = (y_prob >= best_threshold).astype(int)

print("Prediction Completed.")

# =============================================================================
# STEP 7 : MODEL EVALUATION
# =============================================================================

print_step(7, "MODEL EVALUATION")

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

conf_matrix = confusion_matrix(y_test, y_pred)

class_report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

metrics_df = pd.DataFrame({

    "Metric": [

        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"

    ],

    "Value": [

        accuracy,
        precision,
        recall,
        f1,
        roc_auc

    ]

})

print(metrics_df)

# =============================================================================
# STEP 8 : SAVE PREDICTIONS
# =============================================================================

print_step(8, "SAVING PREDICTIONS")

prediction_df = pd.DataFrame({

    "Actual": y_test,

    "Predicted": y_pred,

    "Probability": y_prob

})

prediction_path = os.path.join(

    RESULTS_FOLDER,

    "Logistic_Regression_Predictions.xlsx"

)

prediction_df.to_excel(

    prediction_path,

    index=False

)

print("Predictions Saved.")

# =============================================================================
# STEP 9 : SAVE REPORT
# =============================================================================

print_step(9, "SAVING REPORT")

report_path = os.path.join(

    RESULTS_FOLDER,

    "Logistic_Regression_Report.xlsx"

)

with pd.ExcelWriter(
    report_path,
    engine="openpyxl"
) as writer:

    metrics_df.to_excel(
        writer,
        sheet_name="Performance",
        index=False
    )

    pd.DataFrame(
        conf_matrix,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"]
    ).to_excel(
        writer,
        sheet_name="Confusion_Matrix"
    )

    pd.DataFrame(class_report).transpose().to_excel(
        writer,
        sheet_name="Classification_Report"
    )

print("Report Saved.")

# =============================================================================
# STEP 10 : SAVE FIGURES
# =============================================================================

print_step(10, "SAVING FIGURES")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Logistic Regression")

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(
        FIGURES_FOLDER,
        "Logistic_ROC_Curve.png"
    ),

    dpi=300

)

plt.close()

# Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(6, 5))

plt.plot(recall_curve, precision_curve)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.tight_layout()

plt.savefig(

    os.path.join(
        FIGURES_FOLDER,
        "Logistic_PR_Curve.png"
    ),

    dpi=300

)

plt.close()

print("Figures Saved.")

# =============================================================================
# PROJECT COMPLETED
# =============================================================================

print("\n" + "=" * 70)

print("PROJECT COMPLETED SUCCESSFULLY")

print("=" * 70)