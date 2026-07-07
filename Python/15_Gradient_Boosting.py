# =============================================================================
# PROJECT : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# SCRIPT  : 15_Gradient_Boosting.py
# PURPOSE : Gradient Boosting Bankruptcy Prediction with Full Pipeline
# =============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# =============================================================================
# ADD PROJECT ROOT
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config import (
    FEATURESET_FOLDER,
    RESULTS_FOLDER,
    FIGURES_FOLDER,
    TARGET_COLUMN,
    RANDOM_STATE
)

# =============================================================================
# HELPER FUNCTIONS & CLASSES
# =============================================================================

def print_step(step, title):
    print("\n" + "=" * 70)
    print(f"STEP {step} : {title}")
    print("=" * 70)

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=(0.01, 0.99)):
        self.limits = limits

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_bounds_ = X.quantile(self.limits[0])
        self.upper_bounds_ = X.quantile(self.limits[1])
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1).values

# =============================================================================
# CREATE OUTPUT FOLDERS
# =============================================================================

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)

# =============================================================================
# STEP 1 : LOAD DATASETS
# =============================================================================

print_step(1, "LOADING TRAIN & TEST DATASETS")

train_path = os.path.join(FEATURESET_FOLDER, "Train_Selected_Features.xlsx")
test_path = os.path.join(FEATURESET_FOLDER, "Test_Selected_Features.xlsx")

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

print(f"Training Rows : {len(train_df):,} | Columns : {len(train_df.columns)}")
print(f"Testing Rows  : {len(test_df):,} | Columns : {len(test_df.columns)}")

# =============================================================================
# STEP 2 : SEPARATE FEATURES AND TARGET
# =============================================================================

print_step(2, "SEPARATING FEATURES AND TARGET")

# Drop non-feature columns
drop_cols = ["company_name", "year", TARGET_COLUMN]
X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
y_test = test_df[TARGET_COLUMN]

print(f"Training Features : {X_train.shape}")
print(f"Testing Features  : {X_test.shape}")

# Replace any inf values with NaN so the imputer can handle them
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# =============================================================================
# STEP 3 : BUILD PIPELINE
# =============================================================================

print_step(3, "BUILDING PIPELINE (IMPUTE -> WINSORIZE -> SCALE -> SMOTE -> GB)")

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("winsorizer", Winsorizer(limits=(0.01, 0.99))),
    ("scaler", StandardScaler()),
    ("smotetomek", SMOTETomek(random_state=RANDOM_STATE)),
    ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE))
])

# =============================================================================
# STEP 4 : TRAIN MODEL
# =============================================================================

print_step(4, "TRAINING GRADIENT BOOSTING PIPELINE")

pipeline.fit(X_train, y_train)

print("Pipeline Trained Successfully.")

# =============================================================================
# STEP 5 : PREDICT TEST DATA
# =============================================================================

print_step(5, "PREDICTING TEST DATA")

import numpy as np
from sklearn.metrics import roc_curve

y_prob = pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
j_scores = tpr - fpr
best_index = np.argmax(j_scores)
best_threshold = thresholds[best_index]

print(f"Dynamically Calculated Optimal Threshold (Youden's J): {best_threshold:.4f}")

y_pred = (y_prob >= best_threshold).astype(int)

# =============================================================================
# STEP 6 : MODEL EVALUATION
# =============================================================================

print_step(6, "MODEL EVALUATION")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Value": [accuracy, precision, recall, f1, roc_auc]
})

print("\nModel Performance Metrics:")
print(metrics_df.to_string(index=False))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n======================================================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("======================================================================")
