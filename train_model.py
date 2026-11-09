
from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET = PROJECT_DIR / "diabetes_risk_prediction.csv"
DEFAULT_MODEL = PROJECT_DIR / "model.joblib"


# Load and validate dataset
def load_dataset(
    dataset_path: Path,
    target_column: str
) -> tuple[pd.DataFrame, pd.Series]:

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Please place diabetes_risk_prediction.csv "
            "inside the project folder."
        )

    dataframe = pd.read_csv(
        dataset_path,
        na_values=["?", "NA", "Unknown"]
    )

    # Clean column names
    dataframe.columns = (
        dataframe.columns.astype(str).str.strip()
    )

    if dataframe.columns.duplicated().any():
        raise ValueError("Duplicate column names found.")

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    # Separate target and features
    target = dataframe.pop(target_column)

    valid_rows = target.notna()

    dataframe = dataframe.loc[valid_rows].copy()
    target = target.loc[valid_rows].astype(str).str.strip()

    if target.eq("").any():
        raise ValueError("Empty target labels found.")

    if dataframe.empty or len(dataframe.columns) == 0:
        raise ValueError("Dataset has no usable features.")

    if target.nunique() < 2:
        raise ValueError(
            "Target column must contain at least two classes."
        )

    # Clean feature columns
    for column in dataframe.columns:

        if pd.api.types.is_numeric_dtype(dataframe[column]):
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            )

        else:
            dataframe[column] = (
                dataframe[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

    # Remove completely empty columns
    empty_columns = dataframe.columns[
        dataframe.isna().all()
    ].tolist()

    if empty_columns:
        raise ValueError(
            f"Empty feature columns found: {empty_columns}"
        )

    return dataframe, target


# Build input schema for application
def build_input_schema(
    features: pd.DataFrame
) -> dict:

    schema = {}

    for column in features.columns:

        series = features[column]

        if pd.api.types.is_numeric_dtype(series):

            schema[column] = {
                "kind": "number",
                "integer": pd.api.types.is_integer_dtype(series),
                "default": float(series.median())
            }

        else:

            categories = sorted(
                series.dropna().astype(str).unique().tolist()
            )

            if not categories:
                raise ValueError(
                    f"No categories available for {column}"
                )

            schema[column] = {
                "kind": "category",
                "categories": categories,
                "default": categories[0]
            }

    return schema


# Train machine learning model
def main():

    parser = argparse.ArgumentParser(
        description="Diabetes Risk Prediction Model Training"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET
    )

    parser.add_argument(
        "--target",
        default="class"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL
    )

    args = parser.parse_args()

    print("\nLoading diabetes dataset...")

    features, target = load_dataset(
        args.data,
        args.target
    )

    print("Dataset loaded successfully!")

    print("\nDataset shape:", features.shape)

    # Identify numerical and categorical columns
    numeric_columns = features.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = [
        column for column in features.columns
        if column not in numeric_columns
    ]

    # Numerical preprocessing
    transformers = []

    if numeric_columns:

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    # Categorical preprocessing
    if categorical_columns:

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    # Complete machine learning pipeline
    classifier = Pipeline([
        (
            "preprocessor",
            ColumnTransformer(transformers=transformers)
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ])

    # Validate train/test split
    class_count = target.nunique()

    test_count = math.ceil(len(target) * 0.2)
    train_count = len(target) - test_count

    if (
        target.value_counts().min() < 2
        or test_count < class_count
        or train_count < class_count
    ):
        raise ValueError(
            "Not enough samples for stratified splitting. "
            "Provide more samples for every target class."
        )

    # Split dataset
    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target
    )

    # Train model
    print("\nTraining Logistic Regression model...")

    classifier.fit(X_train, y_train)

    # Predictions
    predictions = classifier.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========== MODEL RESULTS ==========")

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save model and metadata
    args.model.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model_data = {
        "model": classifier,
        "features": features.columns.tolist(),
        "target": args.target,
        "input_schema": build_input_schema(features),
        "classes": classifier.classes_.tolist()
    }

    joblib.dump(
        model_data,
        args.model
    )

    print("\n===================================")

    print("Model trained successfully!")

    print(f"Model saved at: {args.model}")

    print("===================================\n")


if __name__ == "__main__":
    main()
