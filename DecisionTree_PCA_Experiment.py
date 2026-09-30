import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# 1. LOAD DATASETS
# ==========================================

print("Loading datasets...")

df1 = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv")
df2 = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")
df3 = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")

# Combine all datasets
df = pd.concat([df1, df2, df3], ignore_index=True)

print("Dataset loaded successfully.")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

# ==========================================
# 2. CLEAN DATA
# ==========================================

df.columns = df.columns.str.strip()

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Convert features to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values
X = X.fillna(X.mean())

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

print("Original number of features:", X.shape[1])

# ==========================================
# 3. EXPERIMENT SETTINGS
# ==========================================

test_sizes = [0.2, 0.4, 0.6]

pca_components_list = [2, 4, 6, 8, 10]

results = []

# ==========================================
# 4. RUN EXPERIMENTS
# ==========================================

for test_size in test_sizes:

    print("\n======================================")
    print("TEST SIZE:", test_size)
    print("TRAIN SIZE:", 1 - test_size)
    print("======================================")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=0,
        stratify=y
    )

    for n_components in pca_components_list:

        print("\nPCA Components:", n_components)

        # PCA
        pca = PCA(n_components=n_components)

        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Decision Tree
        model = DecisionTreeClassifier(
            criterion="entropy",
            random_state=0
        )

        model.fit(X_train_pca, y_train)

        # Prediction
        y_pred = model.predict(X_test_pca)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        explained_variance = pca.explained_variance_ratio_.sum()

        print("Accuracy :", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall   :", round(recall, 4))
        print("F1 Score :", round(f1, 4))
        print("Explained Variance:", round(explained_variance, 4))

        # Save results
        results.append({
            "Algorithm": "Decision Tree",
            "Test Size": test_size,
            "Train Size": 1 - test_size,
            "PCA Components": n_components,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Explained Variance": explained_variance
        })

# ==========================================
# 5. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_excel(
    "DecisionTree_PCA_Results.xlsx",
    index=False
)

print("\n======================================")
print("EXPERIMENT COMPLETED")
print("======================================")

print("\nResults saved as:")
print("DecisionTree_PCA_Results.xlsx")