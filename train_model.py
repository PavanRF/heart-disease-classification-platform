"""Leakage-safe training and evaluation for the Heart Disease Prediction project."""
from __future__ import annotations
import json
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"

def build_pipeline(model):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])
    return Pipeline([("preprocessor", prep), ("model", model)])

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    required = set(NUMERIC + CATEGORICAL + [TARGET])
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if not set(raw[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("The target column must contain binary labels 0 and 1.")

    before = len(raw)
    data = raw.drop_duplicates().copy()
    duplicates_removed = before - len(data)
    data = data.dropna(subset=[TARGET])
    X, y = data[NUMERIC + CATEGORICAL], data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.05, max_depth=2, random_state=42
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = build_pipeline(estimator)
        cv_out = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "precision": "precision",
                     "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"},
            n_jobs=-1, error_score="raise"
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        result = {
            "model": name,
            "cv_accuracy_mean": float(cv_out["test_accuracy"].mean()),
            "cv_accuracy_std": float(cv_out["test_accuracy"].std()),
            "cv_f1_mean": float(cv_out["test_f1"].mean()),
            "cv_roc_auc_mean": float(cv_out["test_roc_auc"].mean()),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_precision": float(precision_score(y_test, pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, proba))
        }
        rows.append(result)
        fitted[name] = (pipe, pred, proba)
        print(f"{name:20s} | CV accuracy {result['cv_accuracy_mean']:.3f} ± {result['cv_accuracy_std']:.3f} "
              f"| Test accuracy {result['test_accuracy']:.3f} | F1 {result['test_f1']:.3f} "
              f"| ROC-AUC {result['test_roc_auc']:.3f}")

    results = pd.DataFrame(rows).sort_values(["cv_roc_auc_mean", "cv_f1_mean"], ascending=False)
    # Select using cross-validation metrics, not the held-out test set.
    best_name = str(results.iloc[0]["model"])
    best_pipe, best_pred, best_proba = fitted[best_name]
    joblib.dump(best_pipe, MODEL_DIR / "heart_disease_pipeline.joblib")

    # Save transparent evaluation artifacts.
    results.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    (MODEL_DIR / "evaluation.json").write_text(json.dumps({
        "dataset_rows_original": int(before),
        "duplicate_rows_removed": int(duplicates_removed),
        "rows_after_deduplication": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_class_definition": "target=1 in the supplied dataset",
        "selection_rule": "Highest mean 5-fold training cross-validation ROC-AUC, tie-break on CV F1",
        "best_model": best_name,
        "best_model_test_metrics": rows[[r["model"] for r in rows].index(best_name)],
        "classification_report": classification_report(y_test, best_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, best_pred).tolist()
    }, indent=2), encoding="utf-8")

    disp = ConfusionMatrixDisplay.from_predictions(y_test, best_pred, display_labels=["Target 0", "Target 1"], cmap="Blues")
    disp.ax_.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, best_proba)
    plt.title(f"ROC Curve — {best_name}")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "roc_curve.png", dpi=160)
    plt.close()

    # Store the split's labels/probabilities for reproducible review plots.
    pd.DataFrame({"actual_target": y_test.to_numpy(), "predicted_target": best_pred,
                  "predicted_probability_target_1": best_proba}).to_csv(
                      MODEL_DIR / "test_predictions.csv", index=False)

    print("\n--- Dataset / split summary ---")
    print(f"Original rows: {before}; duplicate rows removed: {duplicates_removed}; unique rows: {len(data)}")
    print(f"Train rows: {len(X_train)}; test rows: {len(X_test)}")
    print(f"Selected model (CV only): {best_name}")
    print(f"Saved model: {MODEL_DIR / 'heart_disease_pipeline.joblib'}")
    print(f"Saved comparison: {MODEL_DIR / 'model_comparison.csv'}")
    print("\nIMPORTANT: This is an educational prototype, not a medical diagnostic tool.")

if __name__ == "__main__":
    main()
