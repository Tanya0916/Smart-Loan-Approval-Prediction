import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("loan_prediction.csv")

print("\n===================================")
print("Dataset Loaded Successfully!")
print("===================================\n")

print(df.head())

# ==========================================================
# REMOVE LOAN ID
# ==========================================================

if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# ==========================================================
# REMOVE MISSING TARGET
# ==========================================================

df = df.dropna(subset=["Loan_Status"])

# ==========================================================
# ENCODE TARGET
# ==========================================================

df["Loan_Status"] = df["Loan_Status"].map({
    "Y": 1,
    "N": 0
}).astype(int)

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ==========================================================
# NUMERIC FEATURES
# ==========================================================

numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

# ==========================================================
# CATEGORICAL FEATURES
# ==========================================================

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

# ==========================================================
# NUMERIC PIPELINE
# ==========================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# ==========================================================
# CATEGORICAL PIPELINE
# ==========================================================

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# ==========================================================
# PREPROCESSOR
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# MODELS
# ==========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier()

}

best_pipeline = None
best_model_name = ""
best_accuracy = 0

print("\n===================================")
print("Training Models")
print("===================================\n")

# ==========================================================
# TRAIN ALL MODELS
# ==========================================================

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    prediction = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    print(f"{name} : {accuracy:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_pipeline = pipeline
        best_model_name = name

# ==========================================================
# BEST MODEL
# ==========================================================

print("\n===================================")
print("Best Model")
print("===================================")

print(best_model_name)

print(f"\nAccuracy : {best_accuracy*100:.2f}%")

# ==========================================================
# EVALUATION
# ==========================================================

prediction = best_pipeline.predict(X_test)

print("\n===================================")
print("Classification Report")
print("===================================\n")

print(classification_report(
    y_test,
    prediction
))

print("\n===================================")
print("Confusion Matrix")
print("===================================\n")

print(confusion_matrix(
    y_test,
    prediction
))

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    best_pipeline,
    "model.pkl"
)

print("\n===================================")
print("Model Saved Successfully!")
print("Saved File : model.pkl")
print("===================================")