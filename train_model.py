import json
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = os.environ.get("DATA_PATH", "data/student-mat.csv")
ARTIFACT_DIR = "artifacts"
TARGET = "G3"

# Prediction scenario: estimate final math grade before the final grade is known.
# G1 and G2 are intentionally excluded to avoid using prior-period grades that
# may not be available in an early-intervention scenario.
NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def build_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES)
    ])

def metrics(y_true, y_pred):
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 3)
    }

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download the UCI Student Performance "
            "ZIP, extract student-mat.csv, and place it in the data/ folder. "
            "See README.md for the official download link."
        )

    df = pd.read_csv(DATA_PATH, sep=";")
    df = df.drop_duplicates()
    required = FEATURES + [TARGET]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df[required].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].between(0, 20)]
    if len(df) < 50:
        raise ValueError("Dataset contains too few valid rows after cleaning.")

    X = df[FEATURES]
    y = df[TARGET]

    # Hold out the test set before model selection.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    # 25% of the remaining 80% => 20% overall validation, 60% training.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42
    )

    estimators = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42
        )
    }

    models = {}
    validation_results = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        validation_results[name] = metrics(y_val, pipeline.predict(X_val))
        models[name] = pipeline

    # Select only on validation MAE.
    best_name = min(validation_results, key=lambda name: validation_results[name]["MAE"])
    best_model = models[best_name]

    # Refit selected pipeline on train + validation data, evaluate test once.
    best_model.fit(X_trainval, y_trainval)
    test_results = metrics(y_test, best_model.predict(X_test))

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "model.joblib"))

    sample = df.iloc[0]
    input_schema = []
    for feature in FEATURES:
        if feature in NUMERIC_FEATURES:
            vals = pd.to_numeric(df[feature], errors="coerce").dropna()
            if feature in ("Medu", "Fedu", "traveltime", "studytime", "failures",
                           "famrel", "freetime", "goout", "Dalc", "Walc", "health"):
                minimum, maximum, step = int(vals.min()), int(vals.max()), 1
            else:
                minimum, maximum, step = float(vals.min()), float(vals.max()), 1
            input_schema.append({
                "name": feature, "type": "number", "min": minimum,
                "max": maximum, "step": step
            })
        else:
            choices = sorted(df[feature].dropna().astype(str).unique().tolist())
            input_schema.append({
                "name": feature, "type": "select", "choices": choices
            })

    metadata = {
        "project": "Student Exam Performance Prediction",
        "dataset": "UCI Student Performance - Mathematics (student-mat.csv)",
        "dataset_source": "https://archive.ics.uci.edu/dataset/320/student-performance",
        "target": TARGET,
        "target_description": "Final mathematics grade, scale 0 to 20",
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "input_schema": input_schema,
        "model_name": best_name,
        "validation_results": validation_results,
        "test_results": test_results,
        "dataset_rows_after_cleaning": int(len(df)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "split": "60% train / 20% validation / 20% test",
        "random_state": 42,
        "leakage_note": "G1 and G2 were excluded because this project models an early prediction scenario."
    }
    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nValidation comparison (lower MAE/RMSE is better; higher R2 is generally better):")
    for name, result in validation_results.items():
        print(f"{name:20s} {result}")
    print(f"\nSelected model: {best_name}")
    print(f"Held-out test metrics: {test_results}")
    print(f"Saved model and metadata to ./{ARTIFACT_DIR}/")

if __name__ == "__main__":
    main()
