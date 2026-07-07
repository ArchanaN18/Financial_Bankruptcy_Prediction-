import os
import pandas as pd
import joblib
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PROJECT_ROOT, FEATURESET_FOLDER, RESULTS_FOLDER, FIGURES_FOLDER

def print_step(step, title):
    print("\n" + "=" * 70)
    print(f"STEP {step} : {title}")
    print("=" * 70)

# =============================================================================
# STEP 1 : AGGREGATE FINAL THESIS TABLES
# =============================================================================

print_step(1, "AGGREGATE FINAL THESIS TABLES")

# We will collect the Model Comparison and the SHAP Report and combine them into one Excel file with multiple sheets.
comparison_path = os.path.join(RESULTS_FOLDER, "Model_Comparison.xlsx")
shap_path = os.path.join(RESULTS_FOLDER, "SHAP_Report.xlsx")

if not os.path.exists(comparison_path) or not os.path.exists(shap_path):
    print("ERROR: Run 17_SHAP.py and 18_Model_Comparison.py first!")
    exit(1)

comp_df = pd.read_excel(comparison_path)
shap_df = pd.read_excel(shap_path)

# Also get the list of selected features used in the final model
test_df = pd.read_excel(os.path.join(FEATURESET_FOLDER, "Test_Selected_Features.xlsx"))
selected_features = test_df.columns.tolist()
from config import TARGET_COLUMN
if TARGET_COLUMN in selected_features:
    selected_features.remove(TARGET_COLUMN)
features_df = pd.DataFrame({"Final_Selected_Features": selected_features})

# Save to a single Final_Report.xlsx with sheets
final_report_path = os.path.join(RESULTS_FOLDER, "Final_Report.xlsx")
with pd.ExcelWriter(final_report_path) as writer:
    comp_df.to_excel(writer, sheet_name="Model_Performance", index=False)
    shap_df.to_excel(writer, sheet_name="SHAP_Importance", index=False)
    features_df.to_excel(writer, sheet_name="Selected_Features", index=False)

print(f"Aggregated Final Report generated at {final_report_path}")

print("\n======================================================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("======================================================================")
