import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Load CSV directly from Hugging Face

df = pd.read_csv("AIML Dataset.csv")

# Feature engineering
df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud", "step"], inplace=True)

# Prepare X, y
X = df.drop("isFraud", axis=1)
y = df["isFraud"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y)

# Define pipeline
numeric = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "balanceDiffOrig", "balanceDiffDest"]
categorical = ["type"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(drop="first"), categorical)
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

# Train and save
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "fraud_detect_pipeline.pkl")

