"""Titanic training: Steps 5–7b and blend (portfolio final: --step blend)."""

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
    LEADERBOARD_RF_PARAMS as GEEKY_RF_PARAMS,
    build_geeky837b_matrices,
    cross_validate_leaderboard,
    predict_leaderboard_probabilities,
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
BEST_STEP = "blend"
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


def predict_step5_survival_probs(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    params: dict[str, object] | None = None,
) -> np.ndarray:
    x_train_fe, x_test_fe = engineer_kaggle815(x_train, x_test, y_train)
    model = CatBoostClassifier(**(params or STEP5_PARAMS))
    model.fit(x_train_fe, y_train)
    return model.predict_proba(x_test_fe)[:, 1]


def predict_step7b_survival_probs(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_train, y_train, x_test, _, _ = build_geeky837b_matrices(train, test)
    return predict_leaderboard_probabilities(x_train, y_train, x_test)


def cross_validate_blend(train: pd.DataFrame, test: pd.DataFrame) -> list[float]:
    """OOF blend CV: mean of Step 5 CatBoost and Step 7b RF val probabilities."""
    y_train = train[TARGET_COLUMN]
    feature_cols = [c for c in train.columns if c != TARGET_COLUMN]
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    oof_probs = np.zeros(len(train))
    scores: list[float] = []

    for fold, (trn_idx, val_idx) in enumerate(cv.split(train, y_train), 1):
        train_fold = train.iloc[trn_idx]
        val_fold = train.iloc[val_idx]
        x_tr = train_fold[feature_cols]
        y_tr = train_fold[TARGET_COLUMN]
        x_val = val_fold[feature_cols]
        y_val = val_fold[TARGET_COLUMN]

        p5 = predict_step5_survival_probs(x_tr, y_tr, x_val)

        x_tr_7b, y_tr_7b, x_val_7b, _, _ = build_geeky837b_matrices(train_fold, val_fold)
        model_7b = RandomForestClassifier(**GEEKY_RF_PARAMS)
        model_7b.fit(x_tr_7b, y_tr_7b)
        p7b = model_7b.predict_proba(x_val_7b)[:, 1]

        blend_probs = (p5 + p7b) / 2
        oof_probs[val_idx] = blend_probs
        fold_preds = (blend_probs >= 0.5).astype(int)
        scores.append(float(accuracy_score(y_val, fold_preds)))
        print(f"  Fold {fold} blend CV: {scores[-1]:.4f}")

    return scores


def run_blend(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print("Final blend: average survival probability (Step 5 CatBoost + Step 7b RF)")

    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    x_test = test[FEATURE_COLUMNS]
    test_ids = test["PassengerId"]

    print("Blend OOF CV (5-fold, aligned folds):")
    scores = cross_validate_blend(train, test)
    scores_arr = pd.Series(scores)
    print(f"CV accuracy ({CV_FOLDS}-fold): {scores_arr.round(4).tolist()}")
    print(f"CV mean: {scores_arr.mean():.4f} (+/- {scores_arr.std():.4f})")

    p5 = predict_step5_survival_probs(x_train, y_train, x_test)
    p7b = predict_step7b_survival_probs(train, test)
    blend_probs = (p5 + p7b) / 2
    predictions = (blend_probs >= 0.5).astype(int)

    pred5 = (p5 >= 0.5).astype(int)
    pred7b = (p7b >= 0.5).astype(int)
    print(f"Vs step5 hard labels: {(pred5 == predictions).sum()}/{len(predictions)} agree")
    print(f"Vs step7b hard labels: {(pred7b == predictions).sum()}/{len(predictions)} agree")

    _write_submission(test_ids, predictions, step="blend")
    print(f"Mean blend P(survive): {blend_probs.mean():.3f}")
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


def _submission_path(step: int | str) -> Path:
    name = "submission_step_blend.csv" if step == "blend" else f"submission_step{step}.csv"
    return Path(__file__).resolve().parent / name


def _write_submission(
    test_ids: pd.Series, predictions: np.ndarray, step: int | str
) -> None:
    submission = pd.DataFrame({"PassengerId": test_ids, "Survived": predictions})
    out_path = _submission_path(step)
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Titanic survival model")
    parser.add_argument(
        "--step",
        type=str,
        choices=["5", "6", "7", "7b", "blend"],
        default=BEST_STEP,
        help="blend=final (default), 5=Kaggle815, 6=Optuna, 7=Geeky loose, 7b=Geeky strict",
    )
    args = parser.parse_args()
    step = args.step

    if step in ("7b", "blend"):
        train = pd.read_csv(DATA_DIR / "train.csv")
        test = pd.read_csv(DATA_DIR / "test.csv")
        if step == "7b":
            run_step7b(train, test)
        else:
            run_blend(train, test)
        return

    x_train, y_train, x_test, test_ids = load_data()
    if step == "7":
        run_step7(x_train, y_train, x_test, test_ids)
    else:
        run_step5_or_6(int(step), x_train, y_train, x_test, test_ids)


if __name__ == "__main__":
    main()
