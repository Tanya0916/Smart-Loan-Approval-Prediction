# train_model.py (Part 1A)

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------
# Load Dataset
# ---------------------------------------

df = pd.read_csv("loan_prediction.csv")

print("=" * 60)
print("Dataset Shape:", df.shape)
print("=" * 60)

# ---------------------------------------
# Drop Loan_ID if present
# ---------------------------------------

if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)

# ---------------------------------------
# Target Encoding
# ---------------------------------------

df["Loan_Status"] = df["Loan_Status"].map({
    "Y": 1,
    "N": 0
})

# ---------------------------------------
# Features & Target
# ---------------------------------------

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ---------------------------------------
# Identify Numeric & Categorical Columns
# ---------------------------------------

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numeric Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

# ---------------------------------------
# Numeric Pipeline
# ---------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# ---------------------------------------
# Categorical Pipeline
# ---------------------------------------

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

# ---------------------------------------
# Column Transformer
# ---------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# ---------------------------------------
# Train-Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# ---------------------------------------
# Models
# ---------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

best_model = None
best_accuracy = 0
best_name = ""

print("\nTraining models...\n")
# ---------------------------------------
# Train All Models
# ---------------------------------------

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_name = name

# ---------------------------------------
# Accuracy Comparison
# ---------------------------------------

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for model_name, score in results.items():
    print(f"{model_name:<25} : {score:.4f}")

print("\nBest Model :", best_name)
print("Best Accuracy :", round(best_accuracy * 100, 2), "%")

# ---------------------------------------
# Save Model
# ---------------------------------------

with open("model.joblib", "wb") as file:
    joblib.dump(best_model, file)

print("\nmodel.joblib saved successfully.")

# ---------------------------------------
# Save Accuracy
# ---------------------------------------

accuracy_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": list(results.values())
})

accuracy_df.to_csv(
    "model_accuracy.csv",
    index=False
)

print("model_accuracy.csv saved.")

# ---------------------------------------
# Example Prediction
# ---------------------------------------

sample = X.iloc[[0]]

prediction = best_model.predict(sample)[0]
probability = best_model.predict_proba(sample)[0]

print("\nExample Prediction")
print("----------------------")
print("Prediction :", "Approved" if prediction == 1 else "Rejected")
print("Probability :", probability)

print("\nTraining Completed Successfully.")