import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("loan_prediction.csv")

print("\nDataset Loaded Successfully\n")

print(df.head())

# =====================================================
# DROP ID COLUMN
# =====================================================

df.drop("Loan_ID", axis=1, inplace=True)

# =====================================================
# TARGET COLUMN
# =====================================================

df["Loan_Status"] = df["Loan_Status"].map({
    "Y":1,
    "N":0
})

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# =====================================================
# NUMERIC & CATEGORICAL COLUMNS
# =====================================================

numeric_features = [

    "ApplicantIncome",

    "CoapplicantIncome",

    "LoanAmount",

    "Loan_Amount_Term",

    "Credit_History"

]

categorical_features = [

    "Gender",

    "Married",

    "Dependents",

    "Education",

    "Self_Employed",

    "Property_Area"

]

# =====================================================
# NUMERIC PIPELINE
# =====================================================

numeric_transformer = Pipeline(

    steps=[

        ("imputer",
         SimpleImputer(strategy="median")),

        ("scaler",
         StandardScaler())

    ]

)

# =====================================================
# CATEGORICAL PIPELINE
# =====================================================

categorical_transformer = Pipeline(

    steps=[

        ("imputer",
         SimpleImputer(strategy="most_frequent")),

        ("encoder",
         OneHotEncoder(handle_unknown="ignore"))

    ]

)

# =====================================================
# PREPROCESSOR
# =====================================================

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

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# =====================================================
# MODELS
# =====================================================

models = {

    "Logistic Regression":

        LogisticRegression(max_iter=1000),

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

best_model = None

best_pipeline = None

best_accuracy = 0

print("\n==============================")

print("Training Models")

print("==============================")

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

        best_model = model

        best_pipeline = pipeline
        print("\n==============================")

print("Best Model")

print("==============================")

print(best_model)

print("\nAccuracy : ", round(best_accuracy*100,2),"%")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

prediction = best_pipeline.predict(X_test)

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

        prediction

    )

)

print("\nConfusion Matrix\n")

print(

    confusion_matrix(

        y_test,

        prediction

    )

)

# =====================================================
# SAVE PIPELINE
# =====================================================

joblib.dump(

    best_pipeline,

    "model.pkl"

)

print("\nModel Saved Successfully")

print("File : model.pkl")