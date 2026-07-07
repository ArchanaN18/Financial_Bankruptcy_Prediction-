import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
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

# =============================================================================
# STEP 1 : LOAD DATA AND MODEL
# =============================================================================

print_step(1, "LOAD TEST DATA AND PREPROCESSOR")

test_df = pd.read_excel(os.path.join(FEATURESET_FOLDER, "Test_Selected_Features.xlsx"))
X_test = test_df.drop(columns=[TARGET_COLUMN, "company_name", "year"], errors='ignore')

# Get feature names
feature_names = X_test.columns.tolist()

preprocessor = joblib.load(os.path.join(PROJECT_ROOT, "Models", "Preprocessor.pkl"))
print("Transforming Test Data (Imputation & Scaling, SMOTE skipped)...")
X_test_prep = preprocessor[:-1].transform(X_test)

# Convert back to DataFrame for SHAP
X_test_prep_df = pd.DataFrame(X_test_prep, columns=feature_names)

print("Loading Random Forest Model for SHAP Explainability...")
# We use Random Forest because TreeExplainer works great with it natively
best_model = joblib.load(os.path.join(PROJECT_ROOT, "Models", "BaseModel_rf.pkl"))

# =============================================================================
# STEP 2 : GENERATE SHAP VALUES
# =============================================================================

print_step(2, "GENERATE SHAP VALUES")

print("Initializing SHAP TreeExplainer...")
# TreeExplainer is used for tree-based models
explainer = shap.TreeExplainer(best_model)

# For performance, if the dataset is large, we take a random sample
sample_size = min(2000, len(X_test_prep_df))
print(f"Calculating SHAP values for a random sample of {sample_size} test instances...")
X_sample = X_test_prep_df.sample(n=sample_size, random_state=42)

shap_values = explainer.shap_values(X_sample)

# Random Forest in SHAP might return a list for each class. We want class 1 (bankrupt).
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    # shape is (n_samples, n_features, n_classes)
    shap_values = np.array(shap_values)[:, :, 1]
else:
    shap_values = np.array(shap_values)

# =============================================================================
# STEP 3 : GLOBAL FEATURE IMPORTANCE (SHAP_Report.xlsx)
# =============================================================================

print_step(3, "GENERATE SHAP EXCEL REPORT")

# Calculate mean absolute SHAP value for each feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_report_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP_Value": mean_abs_shap
}).sort_values(by="Mean_Absolute_SHAP_Value", ascending=False)

shap_report_df.to_excel(os.path.join(RESULTS_FOLDER, "SHAP_Report.xlsx"), index=False)
print("Saved SHAP feature importance to Results/SHAP_Report.xlsx")

# =============================================================================
# STEP 4 : SHAP PLOTS
# =============================================================================

print_step(4, "GENERATE SHAP PLOTS")

# 1. SHAP Summary Plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Summary Plot")
plt.savefig(os.path.join(FIGURES_FOLDER, "SHAP_Summary_Plot.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved SHAP Summary Plot.")

# 2. SHAP Bar Plot (Global Importance)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("SHAP Global Feature Importance")
plt.savefig(os.path.join(FIGURES_FOLDER, "SHAP_Bar_Plot.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved SHAP Bar Plot.")

# 3. SHAP Dependence Plot for the Top Feature
top_feature = shap_report_df.iloc[0]["Feature"]
plt.figure(figsize=(8, 6))
shap.dependence_plot(top_feature, shap_values, X_sample, show=False)
plt.title(f"SHAP Dependence Plot: {top_feature}")
plt.savefig(os.path.join(FIGURES_FOLDER, f"SHAP_Dependence_{top_feature}.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved SHAP Dependence Plot.")

print("\n======================================================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("======================================================================")
