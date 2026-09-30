import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier


# =========================
# 1. LOAD DATASETS
# =========================

files = [
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
]

dataframes = []

for file in files:
    print(f"Loading {file}...")
    df = pd.read_csv(file, low_memory=False)
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

print("\nDataset shape:", data.shape)


# =========================
# 2. CLEAN COLUMN NAMES
# =========================

data.columns = data.columns.str.strip()

X = data.iloc[:, :-1].copy()
y = data.iloc[:, -1].copy()


# =========================
# 3. CONVERT FEATURES TO NUMERIC
# =========================

X = X.apply(pd.to_numeric, errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.mean())


# =========================
# 4. ENCODE TARGET
# =========================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


# =========================
# 5. SCALE FEATURES
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================
# 6. EXPERIMENT SETTINGS
# =========================

test_sizes = [0.2, 0.4, 0.6]

pca_components = [2, 4, 6, 8, 10]

results = []


# =========================
# 7. RUN EXPERIMENTS
# =========================

for test_size in test_sizes:

    print("\n====================================")
    print(f"TEST SIZE: {test_size}")
    print("====================================")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=0,
        stratify=y
    )

    for n_components in pca_components:

        print(f"\nRunning PCA with {n_components} components...")

        # PCA
        pca = PCA(n_components=n_components)

        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        explained_variance = pca.explained_variance_ratio_.sum()

        # XGBoost
        model = XGBClassifier(
            random_state=0,
            eval_metric="mlogloss"
        )

        model.fit(X_train_pca, y_train)

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

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"Explained Variance : {explained_variance:.4f}")

        results.append({
            "Algorithm": "XGBoost",
            "Test Size": test_size,
            "PCA Components": n_components,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Explained Variance": explained_variance
        })


# =========================
# 8. SAVE RESULTS
# =========================

results_df = pd.DataFrame(results)

results_df.to_excel(
    "XGBoost_PCA_Results.xlsx",
    index=False
)

print("\n====================================")
print("XGBoost experiment completed!")
print("Results saved as XGBoost_PCA_Results.xlsx")
print("====================================")