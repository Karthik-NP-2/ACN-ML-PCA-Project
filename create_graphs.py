import pandas as pd
import matplotlib.pyplot as plt

# Read final combined results
df = pd.read_excel("model_performance_comparison.xlsx")

# Remove failed QDA experiments
df = df.dropna(subset=["Accuracy"])

# -----------------------------
# 1. Accuracy
# -----------------------------
plt.figure(figsize=(10, 6))

for algorithm in df["Algorithm"].unique():
    data = df[df["Algorithm"] == algorithm]
    plt.plot(
        data["PCA Components"],
        data["Accuracy"],
        marker="o",
        label=algorithm
    )

plt.xlabel("PCA Components")
plt.ylabel("Accuracy")
plt.title("Accuracy vs PCA Components")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Accuracy_vs_PCA.png", dpi=300)
plt.show()


# -----------------------------
# 2. Precision
# -----------------------------
plt.figure(figsize=(10, 6))

for algorithm in df["Algorithm"].unique():
    data = df[df["Algorithm"] == algorithm]
    plt.plot(
        data["PCA Components"],
        data["Precision"],
        marker="o",
        label=algorithm
    )

plt.xlabel("PCA Components")
plt.ylabel("Precision")
plt.title("Precision vs PCA Components")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Precision_vs_PCA.png", dpi=300)
plt.show()


# -----------------------------
# 3. Recall
# -----------------------------
plt.figure(figsize=(10, 6))

for algorithm in df["Algorithm"].unique():
    data = df[df["Algorithm"] == algorithm]
    plt.plot(
        data["PCA Components"],
        data["Recall"],
        marker="o",
        label=algorithm
    )

plt.xlabel("PCA Components")
plt.ylabel("Recall")
plt.title("Recall vs PCA Components")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Recall_vs_PCA.png", dpi=300)
plt.show()


# -----------------------------
# 4. F1 Score
# -----------------------------
plt.figure(figsize=(10, 6))

for algorithm in df["Algorithm"].unique():
    data = df[df["Algorithm"] == algorithm]
    plt.plot(
        data["PCA Components"],
        data["F1 Score"],
        marker="o",
        label=algorithm
    )

plt.xlabel("PCA Components")
plt.ylabel("F1 Score")
plt.title("F1 Score vs PCA Components")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("F1_vs_PCA.png", dpi=300)
plt.show()


# -----------------------------
# 5. PCA Explained Variance
# -----------------------------
plt.figure(figsize=(10, 6))

# Use unique PCA values and average explained variance
pca_data = df.groupby("PCA Components")["Explained Variance"].mean()

plt.plot(
    pca_data.index,
    pca_data.values,
    marker="o"
)

plt.xlabel("PCA Components")
plt.ylabel("Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig("PCA_Explained_Variance.png", dpi=300)
plt.show()


print("\n====================================")
print("All graphs created successfully!")
print("====================================")
print("Accuracy_vs_PCA.png")
print("Precision_vs_PCA.png")
print("Recall_vs_PCA.png")
print("F1_vs_PCA.png")
print("PCA_Explained_Variance.png")