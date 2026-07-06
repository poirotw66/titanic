"""Titanic Step 5: Kaggle 0.8157 notebook recipe (CatBoost)."""

from __future__ import annotations

from pathlib import Path

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
STEP = 5
CV_FOLDS = 5
RANDOM_STATE = 42

CATBOOST_PARAMS = {
    "verbose": False,
    "thread_count": -1,
    "depth": 4,
    "iterations": 1000,
    "learning_rate": 0.0005,
    "random_seed": RANDOM_STATE,
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

        x_tr_fe, x_val_fe = engineer_kaggle815(x_tr, x_val, y_tr)
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(x_tr_fe, y_tr)
        scores.append(float(accuracy_score(y_val, model.predict(x_val_fe))))

    return scores


def main() -> None:
    x_train, y_train, x_test, test_ids = load_data()

    scores = cross_validate(x_train, y_train)
    scores_arr = pd.Series(scores)
    print(f"CV accuracy ({CV_FOLDS}-fold): {scores_arr.round(4).tolist()}")
    print(f"CV mean: {scores_arr.mean():.4f} (+/- {scores_arr.std():.4f})")

    x_train_fe, x_test_fe = engineer_kaggle815(x_train, x_test, y_train)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(x_train_fe, y_train)
    predictions = model.predict(x_test_fe).astype(int)

    submission = pd.DataFrame(
        {"PassengerId": test_ids, "Survived": predictions}
    )
    out_path = Path(__file__).resolve().parent / f"submission_step{STEP}.csv"
    submission.to_csv(out_path, index=False)
    print(f"Wrote {len(submission)} rows to {out_path}")
    print(f"Predicted survival rate: {predictions.mean():.3f}")


if __name__ == "__main__":
    main()
