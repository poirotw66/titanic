"""Titanic training: Steps 5–7b (Step 5 best LB; Step 7b = strict Geeky ipynb)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from features_geeky837 import engineer_geeky837
from features_geeky837b import (
    build_geeky837b_matrices,
    cross_validate_leaderboard,
    predict_leaderboard_submission,
)
from features_kaggle815 import engineer_kaggle815

DATA_DIR = Path(__file__).resolve().parent / "data"
FEATURE_COLUMNS = [
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]
TARGET_COLUMN = "Survived"
BEST_STEP = 5
CV_FOLDS = 5
RANDOM_STATE = 42
OPTUNA_TRIALS = 50

BASE_CATBOOST_PARAMS = {
    "verbose": False,
    "thread_count": -1,
    "random_seed": RANDOM_STATE,
}

STEP5_PARAMS = {
    **BASE_CATBOOST_PARAMS,
    "depth": 4,
    "iterations": 1000,
    "learning_rate": 0.0005,
}

LEADERBOARD_RF_PARAMS = {
    "criterion": "gini",
    "n_estimators": 1750,
    "max_depth": 7,
    "min_samples_split": 6,
    "min_samples_leaf": 6,
    "max_features": "sqrt",
    "oob_score": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    x_test = test[FEATURE_COLUMNS]
    test_ids = test["PassengerId"]
    return x_train, y_train, x_test, test_ids


def cross_validate_kaggle815(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    catboost_params: dict[str, object],
) -> list[float]:
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scores: list[float] = []

    for train_idx, val_idx in cv.split(x_train, y_train):
        x_tr = x_train.iloc[train_idx]
        x_val = x_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        x_tr_fe, x_val_fe = engineer_kaggle815(x_tr, x_val, y_tr)
        model = CatBoostClassifier(**catboost_params)
        model.fit(x_tr_fe, y_tr)
        scores.append(float(accuracy_score(y_val, model.predict(x_val_fe))))

    return scores


def tune_hyperparameters(
    x_train: pd.DataFrame, y_train: pd.Series, n_trials: int
) -> dict[str, object]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            **BASE_CATBOOST_PARAMS,
            "depth": trial.suggest_int("depth", 3, 6),
            "iterations": trial.suggest_int("iterations", 400, 2000, step=100),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 0.01, log=True
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 15),
        }
        scores = cross_validate_kaggle815(x_train, y_train, params)
        return float(sum(scores) / len(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {**BASE_CATBOOST_PARAMS, **study.best_params}
    print(f"Best CV: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return best


def cross_validate_geeky837(
    x_train: pd.DataFrame, y_train: pd.Series
) -> list[float]:
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scores: list[float] = []

    for train_idx, val_idx in cv.split(x_train, y_train):
        x_tr = x_train.iloc[train_idx]
        x_val = x_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        x_tr_fe, x_val_fe, _ = engineer_geeky837(x_tr, x_val, y_tr)
        model = RandomForestClassifier(**LEADERBOARD_RF_PARAMS)
        model.fit(x_tr_fe, y_tr)
        scores.append(float(accuracy_score(y_val, model.predict(x_val_fe))))

    return scores


def run_step5_or_6(step: int, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, test_ids: pd.Series) -> None:
    if step == 6:
        print(f"Optuna search ({OPTUNA_TRIALS} trials)...")
        params = tune_hyperparameters(x_train, y_train, OPTUNA_TRIALS)
        step5_cv = sum(cross_validate_kaggle815(x_train, y_train, STEP5_PARAMS)) / CV_FOLDS
        print(f"Step 5 baseline CV: {step5_cv:.4f}")
    else:
        params = STEP5_PARAMS
        print("Using Step 5 params (best public LB 0.81578)")

    scores = cross_validate_kaggle815(x_train, y_train, params)
    scores_arr = pd.Series(scores)
    print(f"CV accuracy ({CV_FOLDS}-fold): {scores_arr.round(4).tolist()}")
    print(f"CV mean: {scores_arr.mean():.4f} (+/- {scores_arr.std():.4f})")

    x_train_fe, x_test_fe = engineer_kaggle815(x_train, x_test, y_train)
    model = CatBoostClassifier(**params)
    model.fit(x_train_fe, y_train)
    predictions = model.predict(x_test_fe).astype(int)
    _write_submission(test_ids, predictions, step)


def run_step7(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, test_ids: pd.Series
) -> None:
    print("Geeky Codes 0.837 recipe: Survival_Rate + leaderboard RF + fold avg probs")

    scores = cross_validate_geeky837(x_train, y_train)
    scores_arr = pd.Series(scores)
    print(f"CV accuracy ({CV_FOLDS}-fold): {scores_arr.round(4).tolist()}")
    print(f"CV mean: {scores_arr.mean():.4f} (+/- {scores_arr.std():.4f})")

    x_train_full, x_test_full, _ = engineer_geeky837(x_train, x_test, y_train)
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    test_probs = np.zeros(len(x_test_full))

    for train_idx, _ in cv.split(x_train_full, y_train):
        model = RandomForestClassifier(**LEADERBOARD_RF_PARAMS)
        model.fit(x_train_full[train_idx], y_train.iloc[train_idx])
        test_probs += model.predict_proba(x_test_full)[:, 1] / CV_FOLDS

    predictions = (test_probs >= 0.5).astype(int)
    _write_submission(test_ids, predictions, step=7)
    print(f"Predicted survival rate: {predictions.mean():.3f}")


def run_step7b(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("Geeky 0.837 strict ipynb port: separate scalers/encoders, skf random_state=5")

    x_train, y_train, x_test, test_ids, feature_names = build_geeky837b_matrices(
        train, test
    )
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"X_train {x_train.shape}, X_test {x_test.shape}")

    scores = cross_validate_leaderboard(x_train, y_train)
    scores_arr = pd.Series(scores)
    print(f"CV accuracy ({CV_FOLDS}-fold): {scores_arr.round(4).tolist()}")
    print(f"CV mean: {scores_arr.mean():.4f} (+/- {scores_arr.std():.4f})")

    predictions = predict_leaderboard_submission(x_train, y_train, x_test)
    _write_submission(test_ids, predictions, step="7b")
    print(f"Predicted survival rate: {predictions.mean():.3f}")


def _write_submission(
    test_ids: pd.Series, predictions: np.ndarray, step: int | str
) -> None:
    submission = pd.DataFrame({"PassengerId": test_ids, "Survived": predictions})
    out_path = Path(__file__).resolve().parent / f"submission_step{step}.csv"
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Titanic survival model")
    parser.add_argument(
        "--step",
        type=str,
        choices=["5", "6", "7", "7b"],
        default=str(BEST_STEP),
        help="5=Kaggle815 (LB best), 6=Optuna, 7=Geeky837 loose, 7b=Geeky ipynb strict",
    )
    args = parser.parse_args()
    step = args.step

    if step == "7b":
        train = pd.read_csv(DATA_DIR / "train.csv")
        test = pd.read_csv(DATA_DIR / "test.csv")
        run_step7b(train, test)
        return

    x_train, y_train, x_test, test_ids = load_data()
    if step == "7":
        run_step7(x_train, y_train, x_test, test_ids)
    else:
        run_step5_or_6(int(step), x_train, y_train, x_test, test_ids)


if __name__ == "__main__":
    main()
