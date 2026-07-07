import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
)
import joblib
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PROJECT_ROOT, FEATURESET_FOLDER, RESULTS_FOLDER, FIGURES_FOLDER, TARGET_COLUMN
from sklearn.base import BaseEstimator, TransformerMixin

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
# HELPER FUNCTIONS
# =============================================================================

def print_step(step, title):
    print("\n" + "=" * 70)
    print(f"STEP {step} : {title}")
    print("=" * 70)

def generate_report(y_true, y_pred, y_prob, model_name):
    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }
    
    # Save predictions
    preds_df = pd.DataFrame({"Actual": y_true, "Predicted_Class": y_pred, "Predicted_Prob": y_prob})
    preds_df.to_excel(os.path.join(RESULTS_FOLDER, f"{model_name}_Predictions.xlsx"), index=False)
    
    # Save detailed report
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_excel(os.path.join(RESULTS_FOLDER, f"{model_name}_Report.xlsx"))
    
    return metrics

# =============================================================================
# STEP 1 : LOAD TEST DATA AND PREPROCESSOR
# =============================================================================

print_step(1, "LOAD TEST DATA AND PREPROCESSOR")

test_df = pd.read_excel(os.path.join(FEATURESET_FOLDER, "Test_Selected_Features.xlsx"))
X_test = test_df.drop(columns=[TARGET_COLUMN, "company_name", "year"], errors='ignore')
y_test = test_df[TARGET_COLUMN]

preprocessor = joblib.load(os.path.join(PROJECT_ROOT, "Models", "Preprocessor.pkl"))

print("Transforming Test Data (Imputation & Scaling, SMOTE skipped)...")
X_test_prep = preprocessor[:-1].transform(X_test)

# =============================================================================
# STEP 2 : EVALUATE BASE MODELS
# =============================================================================

print_step(2, "EVALUATE BASE MODELS")

model_names = {
    "lr": "Logistic_Regression",
    "xgb": "XGBoost",
    "rf": "Random_Forest",
    "gb": "Gradient_Boosting"
}

all_metrics = []
roc_data = {}
pr_data = {}
base_test_probs = {}

for short_name, full_name in model_names.items():
    print(f"Evaluating {full_name}...")
    model = joblib.load(os.path.join(PROJECT_ROOT, "Models", f"BaseModel_{short_name}.pkl"))
    
    y_prob = model.predict_proba(X_test_prep)[:, 1]
    
    fpr_j, tpr_j, thresholds_j = roc_curve(y_test, y_prob)
    j_scores = tpr_j - fpr_j
    best_threshold = thresholds_j[np.argmax(j_scores)]
    y_pred = (y_prob >= best_threshold).astype(int)
    
    base_test_probs[short_name] = y_prob
    
    metrics = generate_report(y_test, y_pred, y_prob, full_name)
    metrics["Model"] = full_name
    all_metrics.append(metrics)
    
    # Store curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[full_name] = (fpr, tpr, metrics["ROC-AUC"])
    
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec, prec)
    pr_data[full_name] = (rec, prec, pr_auc)

# =============================================================================
# STEP 3 : EVALUATE HFRSE STACKING ENSEMBLE
# =============================================================================

print_step(3, "EVALUATE HFRSE STACKING ENSEMBLE")

meta_learner = joblib.load(os.path.join(PROJECT_ROOT, "Models", "MetaLearner.pkl"))

# Create meta features from base model probabilities
test_meta_features = pd.DataFrame(base_test_probs)

y_prob_meta = meta_learner.predict_proba(test_meta_features)[:, 1]

fpr_j, tpr_j, thresholds_j = roc_curve(y_test, y_prob_meta)
j_scores = tpr_j - fpr_j
best_threshold = thresholds_j[np.argmax(j_scores)]
y_pred_meta = (y_prob_meta >= best_threshold).astype(int)

hfrse_metrics = generate_report(y_test, y_pred_meta, y_prob_meta, "HFRSE_Stacking")
hfrse_metrics["Model"] = "HFRSE_Stacking"
all_metrics.append(hfrse_metrics)

# Store curve data
fpr, tpr, _ = roc_curve(y_test, y_prob_meta)
roc_data["HFRSE_Stacking"] = (fpr, tpr, hfrse_metrics["ROC-AUC"])

prec, rec, _ = precision_recall_curve(y_test, y_prob_meta)
pr_auc = auc(rec, prec)
pr_data["HFRSE_Stacking"] = (rec, prec, pr_auc)

# =============================================================================
# STEP 4 : GENERATE COMPARISON TABLE
# =============================================================================

print_step(4, "GENERATE COMPARISON TABLE")

comparison_df = pd.DataFrame(all_metrics)
# Reorder columns
comparison_df = comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]

print(comparison_df.to_string(index=False))

comparison_df.to_excel(os.path.join(RESULTS_FOLDER, "Model_Comparison.xlsx"), index=False)
print("\nComparison table saved to Results/Model_Comparison.xlsx")

# =============================================================================
# STEP 5 : PLOT ROC AND PR CURVES
# =============================================================================

print_step(5, "PLOT ROC AND PR CURVES")

# ROC Curve
plt.figure(figsize=(10, 8))
for model_name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(FIGURES_FOLDER, "ROC_Comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# PR Curve
plt.figure(figsize=(10, 8))
for model_name, (rec, prec, pr_auc) in pr_data.items():
    plt.plot(rec, prec, label=f'{model_name} (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(FIGURES_FOLDER, "PR_Comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Figures saved to Figures/ folder.")

print("\n======================================================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("======================================================================")
