import pandas as pd

# Result files from the 5 algorithms
files = [
    "DecisionTree_PCA_Results.xlsx",
    "KNN_PCA_Results.xlsx",
    "LogisticRegression_PCA_Results.xlsx",
    "QDA_PCA_Results.xlsx",
    "XGBoost_PCA_Results.xlsx"
]

all_results = []

for file in files:
    print(f"Reading {file}...")
    
    df = pd.read_excel(file)
    all_results.append(df)

# Combine all results
combined = pd.concat(all_results, ignore_index=True)

# Save final Excel file
combined.to_excel(
    "model_performance_comparison.xlsx",
    index=False
)

print("\n====================================")
print("All 5 algorithms combined!")
print("====================================")
print(f"Total experiments: {len(combined)}")
print("Saved as: model_performance_comparison.xlsx")