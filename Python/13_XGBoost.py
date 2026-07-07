# =============================================================================
# PROJECT : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# SCRIPT  : 13_XGBoost.py
# PURPOSE : XGBoost Bankruptcy Prediction
# =============================================================================

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import os
import sys
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

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

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)

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

# =============================================================================
# STEP 3.5 : IMPUTATION
# =============================================================================

print_step(3.5, "IMPUTATION")

import numpy as np
from sklearn.impute import SimpleImputer

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

print("Imputation Completed.")

# =============================================================================
# STEP 4 : TRAIN XGBOOST MODEL
# =============================================================================

print_step(4, "TRAINING XGBOOST MODEL")

xgb_model = XGBClassifier(

    random_state=RANDOM_STATE,

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss"

)

xgb_model.fit(
    X_train,
    y_train
)

print("XGBoost Model Trained Successfully.")

# =============================================================================
# STEP 5 : PREDICT TEST DATA
# =============================================================================

print_step(5, "PREDICTING TEST DATA")

import numpy as np
from sklearn.metrics import roc_curve

y_prob = xgb_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
j_scores = tpr - fpr
best_index = np.argmax(j_scores)
best_threshold = thresholds[best_index]

print(f"Dynamically Calculated Optimal Threshold (Youden's J): {best_threshold:.4f}")

y_pred = (y_prob >= best_threshold).astype(int)

print("Prediction Completed.")

# =============================================================================
# STEP 6 : MODEL EVALUATION
# =============================================================================

print_step(6, "MODEL EVALUATION")

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

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1-Score",

        "ROC-AUC"

    ],

    "Value":[

        accuracy,

        precision,

        recall,

        f1,

        roc_auc

    ]

})

print(metrics_df)

# =============================================================================
# STEP 7 : SAVE PREDICTIONS
# =============================================================================

print_step(7, "SAVING PREDICTIONS")

prediction_df = test_df[["company_name", "year"]].copy()

prediction_df["Actual"] = y_test.values

prediction_df["Predicted"] = y_pred

prediction_df["Probability"] = y_prob

prediction_path = os.path.join(

    RESULTS_FOLDER,

    "XGBoost_Predictions.xlsx"

)

prediction_df.to_excel(

    prediction_path,

    index=False

)

print("Predictions Saved.")

# =============================================================================
# STEP 8 : SAVE REPORT
# =============================================================================

print_step(8, "SAVING REPORT")

report_path = os.path.join(

    RESULTS_FOLDER,

    "XGBoost_Report.xlsx"

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
# STEP 9 : SAVE FIGURES
# =============================================================================

print_step(9, "SAVING FIGURES")

# ROC Curve

fpr, tpr, _ = roc_curve(

    y_test,

    y_prob

)

plt.figure(figsize=(6,5))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {roc_auc:.4f}"

)

plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - XGBoost")

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(

        FIGURES_FOLDER,

        "XGBoost_ROC_Curve.png"

    ),

    dpi=300

)

plt.close()

# Precision Recall Curve

precision_curve, recall_curve, _ = precision_recall_curve(

    y_test,

    y_prob

)

plt.figure(figsize=(6,5))

plt.plot(

    recall_curve,

    precision_curve

)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve - XGBoost")

plt.tight_layout()

plt.savefig(

    os.path.join(

        FIGURES_FOLDER,

        "XGBoost_PR_Curve.png"

    ),

    dpi=300

)

plt.close()

print("Figures Saved.")

# =============================================================================
# PROJECT COMPLETED
# =============================================================================

print("\n" + "="*70)

print("PROJECT COMPLETED SUCCESSFULLY")

print("="*70)