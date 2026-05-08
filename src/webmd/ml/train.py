# ============================================================
# ml/train.py — Train, evaluate, save and load ML models
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from webmd.config import ML_MODEL_PATH, ML_RANDOM_STATE, ML_TEST_SIZE, ML_VAL_RATIO


@dataclass
class MlSplits:
    """Holds the 70/15/15 stratified train/val/test split arrays."""
    X_train: pd.DataFrame
    X_val:   pd.DataFrame
    X_test:  pd.DataFrame
    y_train: pd.Series
    y_val:   pd.Series
    y_test:  pd.Series


@dataclass
class ModelResult:
    """Evaluation results for a single trained model."""
    model:    Any
    y_pred:   np.ndarray
    accuracy: float
    f1:       float
    val_f1:   float
    report:   dict[str, Any]
    cm:       np.ndarray


def split_data(X: pd.DataFrame, y: pd.Series) -> MlSplits:
    """Stratified 70 / 15 / 15 split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=ML_TEST_SIZE, random_state=ML_RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=ML_VAL_RATIO,
        random_state=ML_RANDOM_STATE, stratify=y_temp,
    )
    print(
        f"Split — Train: {len(X_train):,} | "
        f"Val: {len(X_val):,} | Test: {len(X_test):,}"
    )
    return MlSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def build_models() -> dict[str, Any]:
    """Instantiate all four classifiers with their fixed hyperparameters.

    XGBoost note: `use_label_encoder` was removed in XGBoost 2.x — not used here.
    XGBoost expects 0-indexed labels; the shift is handled in train_all().
    """
    from xgboost import XGBClassifier  # heavy import — kept local

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            n_jobs=-1, random_state=ML_RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=ML_RANDOM_STATE,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=500, random_state=ML_RANDOM_STATE,
            )),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss",
            n_jobs=-1, random_state=ML_RANDOM_STATE, verbosity=0,
        ),
    }


def train_all(
    models: dict[str, Any],
    splits: MlSplits,
) -> dict[str, ModelResult]:
    """Train every model and return evaluation results keyed by model name."""
    results: dict[str, ModelResult] = {}

    # XGBoost requires 0-indexed labels (Effectiveness is 1-5)
    y_train_xgb = splits.y_train - 1
    y_val_xgb   = splits.y_val   - 1

    for name, model in models.items():
        print(f"\nTraining {name}...")

        if name == "XGBoost":
            model.fit(
                splits.X_train, y_train_xgb,
                eval_set=[(splits.X_val, y_val_xgb)],
                verbose=False,
            )
            y_pred_val  = model.predict(splits.X_val)  + 1  # shift back to 1-5
            y_pred_test = model.predict(splits.X_test) + 1
        else:
            model.fit(splits.X_train, splits.y_train)
            y_pred_val  = model.predict(splits.X_val)
            y_pred_test = model.predict(splits.X_test)

        val_f1 = f1_score(splits.y_val, y_pred_val, average="weighted")
        acc    = accuracy_score(splits.y_test, y_pred_test)
        f1     = f1_score(splits.y_test, y_pred_test, average="weighted")

        results[name] = ModelResult(
            model=model,
            y_pred=y_pred_test,
            accuracy=acc,
            f1=f1,
            val_f1=val_f1,
            report=classification_report(
                splits.y_test, y_pred_test,
                output_dict=True, zero_division=0,
            ),
            cm=confusion_matrix(splits.y_test, y_pred_test),
        )
        print(
            f"  Val F1: {val_f1:.4f}  |  "
            f"Test Accuracy: {acc:.4f}  |  Test F1: {f1:.4f}"
        )

    return results


def select_best(results: dict[str, ModelResult]) -> str:
    """Return the name of the model with the highest test F1."""
    best = max(results, key=lambda k: results[k].f1)
    print(f"\nBest model: {best} (Test F1={results[best].f1:.4f})")
    return best


def save_model(model: Any, path: Path = ML_MODEL_PATH) -> None:
    """Persist a fitted model to disk with joblib."""
    import joblib  # heavy import — kept local

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to '{path}'")


def load_model(path: Path = ML_MODEL_PATH) -> Any:
    """Load a joblib-serialised model from disk.

    Raises FileNotFoundError with a helpful message if the artifact is missing.
    """
    import joblib  # heavy import — kept local

    if not path.exists():
        raise FileNotFoundError(
            f"ML model not found: {path}\n"
            "Run `uv run webmd-ml` first to train and save it."
        )
    return joblib.load(path)
