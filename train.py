"""Titanic training: Step 5 (best LB) or Step 6 (Optuna experiment)."""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

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

# LB 0.81578 — keep for production submissions
STEP5_PARAMS = {
    **BASE_CATBOOST_PARAMS,
    "depth": 4,
    "iterations": 1000,
    "learning_rate": 0.0005,
}


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    x_test = test[FEATURE_COLUMNS]
    test_ids = test["PassengerId"]
    return x_train, y_train, x_test, test_ids


def cross_validate(
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
        scores = cross_validate(x_train, y_train, params)
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


def run_step(step: int) -> None:
    x_train, y_train, x_test, test_ids = load_data()

    if step == 6:
        print(f"Optuna search ({OPTUNA_TRIALS} trials)...")
        params = tune_hyperparameters(x_train, y_train, OPTUNA_TRIALS)
        step5_cv = sum(cross_validate(x_train, y_train, STEP5_PARAMS)) / CV_FOLDS
        print(f"Step 5 baseline CV: {step5_cv:.4f}")
    else:
        params = STEP5_PARAMS
        print("Using Step 5 params (best public LB 0.81578)")

    scores = cross_validate(x_train, y_train, params)
    scores_arr = pd.Series(scores)
    print(f"CV accuracy ({CV_FOLDS}-fold): {scores_arr.round(4).tolist()}")
    print(f"CV mean: {scores_arr.mean():.4f} (+/- {scores_arr.std():.4f})")

    x_train_fe, x_test_fe = engineer_kaggle815(x_train, x_test, y_train)
    model = CatBoostClassifier(**params)
    model.fit(x_train_fe, y_train)
    predictions = model.predict(x_test_fe).astype(int)

    submission = pd.DataFrame(
        {"PassengerId": test_ids, "Survived": predictions}
    )
    out_path = Path(__file__).resolve().parent / f"submission_step{step}.csv"
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)} rows to {out_path}")
    print(f"Predicted survival rate: {predictions.mean():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Titanic survival model")
    parser.add_argument(
        "--step",
        type=int,
        choices=[5, 6],
        default=BEST_STEP,
        help=f"5=notebook recipe (default, LB best), 6=Optuna experiment",
    )
    args = parser.parse_args()
    run_step(args.step)


if __name__ == "__main__":
    main()
