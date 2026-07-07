# =============================================================================
# VARIABLE MAPPING
# =============================================================================
# Project :
# Hybrid Financial Representation Stacking Ensemble (HFRSE)
#
# Purpose :
# This file stores metadata for all original financial variables
# used in the U.S. Bankruptcy Prediction Dataset.
#
# This file will be used by:
#   • 03_Data_Dictionary.py
#   • 05_Exploratory_Data_Analysis.py
#   • 06_Traditional_Ratios.py
#   • 07_Composition_Features.py
#   • 08_Hybrid_Features.py
#   • 17_SHAP.py
#   • 19_Final_Report.py
# =============================================================================

VARIABLES = {

    "X1": {
        "name": "Current Assets",
        "description": "Assets expected to be converted into cash or consumed within one year.",
        "statement": "Balance Sheet",
        "category": "Current Asset",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X2": {
        "name": "Cost of Goods Sold",
        "description": "Direct costs incurred in producing goods or services sold by the company.",
        "statement": "Income Statement",
        "category": "Expense",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X3": {
        "name": "Depreciation & Amortization",
        "description": "Non-cash expenses representing the allocation of tangible and intangible asset costs over time.",
        "statement": "Income Statement",
        "category": "Expense",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X4": {
        "name": "EBITDA",
        "description": "Earnings before interest, taxes, depreciation and amortization.",
        "statement": "Income Statement",
        "category": "Profitability",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X5": {
        "name": "Inventory",
        "description": "Goods held by the company for sale or production.",
        "statement": "Balance Sheet",
        "category": "Current Asset",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X6": {
        "name": "Net Income",
        "description": "Final profit earned after deducting all operating and non-operating expenses.",
        "statement": "Income Statement",
        "category": "Profitability",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X7": {
        "name": "Total Receivables",
        "description": "Amounts owed to the company by customers or other entities.",
        "statement": "Balance Sheet",
        "category": "Current Asset",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X8": {
        "name": "Market Value",
        "description": "Estimated market value of the company based on market capitalization or valuation.",
        "statement": "Market Information",
        "category": "Market",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X9": {
        "name": "Net Sales",
        "description": "Revenue generated after deducting returns, discounts and allowances.",
        "statement": "Income Statement",
        "category": "Revenue",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X10": {
        "name": "Total Assets",
        "description": "Total value of all assets owned by the company.",
        "statement": "Balance Sheet",
        "category": "Asset",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X11": {
        "name": "Long-Term Debt",
        "description": "Financial obligations that mature after more than one year.",
        "statement": "Balance Sheet",
        "category": "Liability",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X12": {
        "name": "EBIT",
        "description": "Earnings before interest and taxes.",
        "statement": "Income Statement",
        "category": "Profitability",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X13": {
        "name": "Gross Profit",
        "description": "Revenue remaining after deducting the cost of goods sold.",
        "statement": "Income Statement",
        "category": "Profitability",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X14": {
        "name": "Current Liabilities",
        "description": "Financial obligations that are due within one year.",
        "statement": "Balance Sheet",
        "category": "Current Liability",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X15": {
        "name": "Retained Earnings",
        "description": "Accumulated profits retained in the business after dividend payments.",
        "statement": "Balance Sheet",
        "category": "Equity",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X16": {
        "name": "Total Revenue",
        "description": "Total income generated from business operations before deductions.",
        "statement": "Income Statement",
        "category": "Revenue",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X17": {
        "name": "Total Liabilities",
        "description": "Total amount owed by the company to external parties.",
        "statement": "Balance Sheet",
        "category": "Liability",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    },

    "X18": {
        "name": "Operating Expenses",
        "description": "Expenses incurred through normal business operations excluding cost of goods sold.",
        "statement": "Income Statement",
        "category": "Expense",
        "dtype": "float64",
        "unit": "-",
        "feature_group": "Raw Financial Variable"
    }

}