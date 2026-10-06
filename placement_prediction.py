import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. READ DATASET
# =========================================================

df = pd.read_csv("student_placement.csv")

print("========== DATASET INFORMATION ==========")
print("Total records:", len(df))
print("Total columns:", len(df.columns))

print("\nFirst 5 records:")
print(df.head())


# =========================================================
# 2. CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# =========================================================
# 3. ENCODE CATEGORICAL DATA
# =========================================================

le = LabelEncoder()

df["Internship"] = le.fit_transform(df["Internship"])
df["Placement_Status"] = le.fit_transform(df["Placement_Status"])

print("\n========== AFTER ENCODING ==========")
print(df.head())


# =========================================================
# 4. SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop(["Student_ID", "Placement_Status"], axis=1)
y = df["Placement_Status"]

print("\n========== FEATURES ==========")
print("Number of features:", X.shape[1])

print("\n========== TARGET ==========")
print(y.value_counts())


# =========================================================
# 5. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================================================
# 6. NORMALIZATION
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n========== NORMALIZATION COMPLETED ==========")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# =========================================================
# 7. LOGISTIC REGRESSION MODEL
# =========================================================

lr_model = LogisticRegression(random_state=42)

lr_model.fit(X_train, y_train)

lr_prediction = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_prediction)

print("\n========== LOGISTIC REGRESSION ==========")
print("Accuracy:", round(lr_accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_prediction))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        lr_prediction,
        target_names=["Not Placed", "Placed"]
    )
)


# =========================================================
# 8. RANDOM FOREST MODEL
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_prediction = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_prediction)

print("\n========== RANDOM FOREST ==========")
print("Accuracy:", round(rf_accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_prediction))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_prediction,
        target_names=["Not Placed", "Placed"]
    )
)


# =========================================================
# 9. MODEL COMPARISON
# =========================================================

print("\n========== MODEL COMPARISON ==========")

print(
    "Logistic Regression Accuracy:",
    round(lr_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest Accuracy:",
    round(rf_accuracy * 100, 2),
    "%"
)


# =========================================================
# 10. LOGISTIC REGRESSION CONFUSION MATRIX
# =========================================================

cm_lr = confusion_matrix(y_test, lr_prediction)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    xticklabels=["Not Placed", "Placed"],
    yticklabels=["Not Placed", "Placed"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")

plt.tight_layout()
plt.show()


# =========================================================
# 11. RANDOM FOREST CONFUSION MATRIX
# =========================================================

cm_rf = confusion_matrix(y_test, rf_prediction)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    xticklabels=["Not Placed", "Placed"],
    yticklabels=["Not Placed", "Placed"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")

plt.tight_layout()
plt.show()


# =========================================================
# 12. ACCURACY COMPARISON GRAPH
# =========================================================

models = [
    "Logistic Regression",
    "Random Forest"
]

accuracies = [
    lr_accuracy * 100,
    rf_accuracy * 100
]

plt.figure(figsize=(7, 5))

plt.bar(models, accuracies)

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")

plt.ylim(0, 100)

for i, accuracy in enumerate(accuracies):
    plt.text(
        i,
        accuracy + 1,
        f"{accuracy:.1f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()


# =========================================================
# PROJECT COMPLETED
# =========================================================

print("\n========== PROJECT ANALYSIS COMPLETED ==========")

if lr_accuracy > rf_accuracy:
    print("Best Model: Logistic Regression")
elif rf_accuracy > lr_accuracy:
    print("Best Model: Random Forest")
else:
    print("Both models have the same accuracy.")