"""
Baseline model training and evaluation script using TF-IDF vectorization,
Categorical One-Hot Encoding, and Numerical Scaling with Logistic Regression.

Inputs:
  - data/processed/processed_train.csv
  - data/processed/processed_valid.csv
  - data/processed/processed_test.csv

Outputs:
  - artifacts/results.json
"""

import json
import pathlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Resolve repo root path
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
PROCESSED_DIR = _ROOT / "data" / "processed"
ARTIFACTS_DIR = _ROOT / "artifacts"

CATEGORICAL_COLS = ['subject', 'speaker', 'speaker-title', 'state', 'party-affiliation']
NUMERICAL_COLS = ['barely-true', 'false', 'half-true', 'mostly-true', 'pof']
TEXT_COL = 'article'
TARGET_COL = 'label'


def load_data(data_dir: pathlib.Path = PROCESSED_DIR):
    """Load preprocessed train, valid, and test data splits."""
    train_path = data_dir / "processed_train.csv"
    valid_path = data_dir / "processed_valid.csv"
    test_path = data_dir / "processed_test.csv"

    if not (train_path.exists() and valid_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Preprocessed datasets not found in {data_dir}. "
            "Please run prepare_dataset.py first."
        )

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    # Ensure text column has no NaN values
    train_df[TEXT_COL] = train_df[TEXT_COL].fillna("")
    valid_df[TEXT_COL] = valid_df[TEXT_COL].fillna("")
    test_df[TEXT_COL] = test_df[TEXT_COL].fillna("")

    return train_df, valid_df, test_df

# Build the scikit-learn preprocessing and classification pipeline
def build_pipeline():
    preprocessor = ColumnTransformer([
        ('text', TfidfVectorizer(stop_words='english'), TEXT_COL),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS),
        ('num', StandardScaler(), NUMERICAL_COLS)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    return model

# Train baseline Logistic Regression and evaluate on train, valid, and test sets.
def train_and_evaluate():
    print("Loading preprocessed datasets...")
    train_df, valid_df, test_df = load_data()

    feature_cols = [TEXT_COL] + CATEGORICAL_COLS + NUMERICAL_COLS
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_val, y_val = valid_df[feature_cols], valid_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    print(f"Train shape: {X_train.shape}, Valid shape: {X_val.shape}, Test shape: {X_test.shape}")

    print("Building and training Logistic Regression model...")
    model = build_pipeline()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("Evaluation Results\n")
    print(f"Train Accuracy:      {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")

    # Reports
    val_report_str = classification_report(y_val, y_val_pred)
    test_report_str = classification_report(y_test, y_test_pred)

    val_report_dict = classification_report(y_val, y_val_pred, output_dict=True)
    test_report_dict = classification_report(y_test, y_test_pred, output_dict=True)

    val_cm = confusion_matrix(y_val, y_val_pred).tolist()
    test_cm = confusion_matrix(y_test, y_test_pred).tolist()

    print("\nValidation Classification Report\n")
    print(val_report_str)

    print("\nValidation Confusion Matrix\n")
    print(confusion_matrix(y_val, y_val_pred))

    print("\nTest Classification Report\n")
    print(test_report_str)

    print("\nTest Confusion Matrix\n")
    print(confusion_matrix(y_test, y_test_pred))

    # Compile results for JSON
    results = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "classification_report": test_report_dict,
        "confusion_matrix": test_cm,
        "val_classification_report": val_report_dict,
        "val_confusion_matrix": val_cm,
        "test_classification_report": test_report_dict,
        "test_confusion_matrix": test_cm
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = ARTIFACTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved metrics to {results_path}")
    return model, results

if __name__ == "__main__":
    train_and_evaluate()
