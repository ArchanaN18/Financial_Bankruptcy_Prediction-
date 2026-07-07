# =============================================================================
# PROJECT : Hybrid Financial Representation Stacking Ensemble (HFRSE)
# SCRIPT  : 16_Stacking_Ensemble.py
# PURPOSE : Stacking Ensemble Meta-Learner implementation
# =============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

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
    MODELS_FOLDER,
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
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)

# =============================================================================
# STEP 1 : LOAD DATASETS
# =============================================================================

print_step(1, "LOADING TRAIN & TEST DATASETS")

train_path = os.path.join(FEATURESET_FOLDER, "Train_Selected_Features.xlsx")
test_path = os.path.join(FEATURESET_FOLDER, "Test_Selected_Features.xlsx")

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

# =============================================================================
# STEP 2 : PREPARE DATA
# =============================================================================

print_step(2, "PREPARING DATA AND EXTRACTING GROUPS")

groups_train = train_df['company_name']

drop_cols = ["company_name", "year", TARGET_COLUMN]
X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
y_test = test_df[TARGET_COLUMN]

# Replace inf with nan
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# =============================================================================
# STEP 3 : DEFINE BASE MODELS
# =============================================================================

print_step(3, "DEFINING BASE MODELS")

# Calculate ratio of negatives to positives for scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

base_models = {
    "lr": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver="lbfgs", class_weight="balanced"),
    "xgb": xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", scale_pos_weight=scale_pos_weight),
    "rf": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
    "gb": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)
}

# We will store Out-Of-Fold predictions for the training set
oof_preds = {name: np.zeros(len(X_train)) for name in base_models}

# We will store Test predictions for each fold, then average them
test_preds = {name: np.zeros(len(X_test)) for name in base_models}

# =============================================================================
# STEP 4 : STRATIFIED GROUP K-FOLD OOF GENERATION
# =============================================================================

print_step(4, "STRATIFIED GROUP K-FOLD OOF GENERATION (THIS MIGHT TAKE A WHILE)")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups=groups_train)):
    print(f"\n--- FOLD {fold+1} ---")
    
    # Split Data
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # Preprocessing Pipeline (excluding the estimator)
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorizer", Winsorizer(limits=(0.01, 0.99))),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE))
    ])
    
    print("Fitting Preprocessor (Impute -> Winsorize -> Scale -> SMOTE)...")
    X_tr_resampled, y_tr_resampled = preprocessor.fit_resample(X_tr, y_tr)
    
    print("Transforming Validation and Test Data...")
    # Manually apply transform steps on Validation and Test data
    imputer = preprocessor.named_steps["imputer"]
    winsorizer = preprocessor.named_steps["winsorizer"]
    scaler = preprocessor.named_steps["scaler"]
    
    X_va_transformed = scaler.transform(winsorizer.transform(imputer.transform(X_va)))
    X_test_transformed = scaler.transform(winsorizer.transform(imputer.transform(X_test)))
    
    for name, model in base_models.items():
        print(f"Training {name.upper()}...")
        # Train model
        model.fit(X_tr_resampled, y_tr_resampled)
        
        # OOF Predictions (probability of class 1)
        oof_preds[name][val_idx] = model.predict_proba(X_va_transformed)[:, 1]
        
        # Test Predictions (probability of class 1), accumulated
        test_preds[name] += model.predict_proba(X_test_transformed)[:, 1] / sgkf.n_splits

print("\nOOF Predictions Generated Successfully.")

# =============================================================================
# STEP 5 : TRAIN META-LEARNER
# =============================================================================

print_step(5, "TRAINING META-LEARNER")

# Create OOF Features DataFrame
oof_df = pd.DataFrame(oof_preds)

meta_learner = LogisticRegression(random_state=RANDOM_STATE, class_weight="balanced")
meta_learner.fit(oof_df, y_train)

print("Meta-Learner Trained Successfully.")

# =============================================================================
# STEP 6 : FINAL EVALUATION ON TEST SET
# =============================================================================

print_step(6, "FINAL EVALUATION ON TEST SET")

# Create Test Features DataFrame from base model averaged test predictions
test_features_df = pd.DataFrame(test_preds)

import numpy as np
from sklearn.metrics import roc_curve

y_pred_prob = meta_learner.predict_proba(test_features_df)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
j_scores = tpr - fpr
best_index = np.argmax(j_scores)
best_threshold = thresholds[best_index]

print(f"Dynamically Calculated Optimal Threshold (Youden's J): {best_threshold:.4f}")

y_pred = (y_pred_prob >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Value": [accuracy, precision, recall, f1, roc_auc]
})

print("\nStacking Ensemble Performance Metrics:")
print(metrics_df.to_string(index=False))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =============================================================================
# STEP 7 : SERIALIZE MODELS
# =============================================================================

print_step(7, "SERIALIZING MODELS")

# We train final base models on the ENTIRE training set (no validation fold)
print("Retraining Final Base Models on Entire Training Set...")

final_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("winsorizer", Winsorizer(limits=(0.01, 0.99))),
    ("scaler", StandardScaler()),
    ("smotetomek", SMOTETomek(random_state=RANDOM_STATE))
])

X_train_resampled, y_train_resampled = final_preprocessor.fit_resample(X_train, y_train)

for name, model in base_models.items():
    print(f"Retraining {name.upper()}...")
    model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(model, os.path.join(MODELS_FOLDER, f"BaseModel_{name}.pkl"))

joblib.dump(final_preprocessor, os.path.join(MODELS_FOLDER, "Preprocessor.pkl"))
joblib.dump(meta_learner, os.path.join(MODELS_FOLDER, "MetaLearner.pkl"))

print("All models serialized successfully in 'Models' folder.")

print("\n======================================================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("======================================================================")
