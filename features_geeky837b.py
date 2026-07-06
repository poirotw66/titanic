"""Strict port of titanic-advanced-feature-engineering-tutorial.ipynb (Gunes Evitan)."""

from __future__ import annotations

import string
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

SEED = 42
N_FOLDS = 5
TRAIN_END = 890

DROP_COLS = [
    "Deck",
    "Embarked",
    "Family",
    "Family_Size",
    "Family_Size_Grouped",
    "Survived",
    "Name",
    "Parch",
    "PassengerId",
    "Pclass",
    "Sex",
    "SibSp",
    "Ticket",
    "Title",
    "Ticket_Survival_Rate",
    "Family_Survival_Rate",
    "Ticket_Survival_Rate_NA",
    "Family_Survival_Rate_NA",
]

NON_NUMERIC_FEATURES = [
    "Embarked",
    "Sex",
    "Deck",
    "Title",
    "Family_Size_Grouped",
    "Age",
    "Fare",
]

CAT_FEATURES = ["Pclass", "Sex", "Deck", "Embarked", "Title", "Family_Size_Grouped"]

FAMILY_MAP = {
    1: "Alone",
    2: "Small",
    3: "Small",
    4: "Small",
    5: "Medium",
    6: "Medium",
    7: "Large",
    8: "Large",
    11: "Large",
}

LEADERBOARD_RF_PARAMS: dict[str, Any] = {
    "criterion": "gini",
    "n_estimators": 1750,
    "max_depth": 7,
    "min_samples_split": 6,
    "min_samples_leaf": 6,
    "max_features": "sqrt",  # notebook: 'auto' (sqrt for sklearn >=1.1)
    "oob_score": True,
    "random_state": SEED,
    "n_jobs": -1,
}


def concat_df(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)


def divide_df(all_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = all_data.loc[:TRAIN_END].copy()
    test = all_data.loc[TRAIN_END + 1 :].drop(["Survived"], axis=1).copy()
    return train, test


def extract_surname(data: pd.Series) -> list[str]:
    families: list[str] = []
    for i in range(len(data)):
        name = data.iloc[i]
        name_no_bracket = name.split("(")[0] if "(" in name else name
        family = name_no_bracket.split(",")[0]
        for char in string.punctuation:
            family = family.replace(char, "").strip()
        families.append(family)
    return families


def _impute_and_deck(df_all: pd.DataFrame) -> None:
    filled_age = df_all.groupby(["Sex", "Pclass"])["Age"].apply(
        lambda x: x.fillna(x.median())
    )
    if isinstance(filled_age.index, pd.MultiIndex):
        filled_age = filled_age.reset_index(level=[0, 1], drop=True)
    df_all["Age"] = filled_age

    df_all["Embarked"] = df_all["Embarked"].fillna("S")
    med_fare = df_all.groupby(["Pclass", "Parch", "SibSp"]).Fare.median()[3][0][0]
    df_all["Fare"] = df_all["Fare"].fillna(med_fare)

    df_all["Deck"] = df_all["Cabin"].apply(lambda s: s[0] if pd.notnull(s) else "M")
    df_all.loc[df_all["Deck"] == "T", "Deck"] = "A"
    df_all["Deck"] = df_all["Deck"].replace(["A", "B", "C"], "ABC")
    df_all["Deck"] = df_all["Deck"].replace(["D", "E"], "DE")
    df_all["Deck"] = df_all["Deck"].replace(["F", "G"], "FG")
    df_all.drop(["Cabin"], inplace=True, axis=1)


def _feature_engineering(df_all: pd.DataFrame) -> None:
    df_all["Fare"] = pd.qcut(df_all["Fare"], 13, duplicates="drop")
    df_all["Age"] = pd.qcut(df_all["Age"], 10, duplicates="drop")
    df_all["Family_Size"] = df_all["SibSp"] + df_all["Parch"] + 1
    df_all["Family_Size_Grouped"] = df_all["Family_Size"].map(FAMILY_MAP)
    df_all["Ticket_Frequency"] = df_all.groupby("Ticket")["Ticket"].transform("count")

    df_all["Title"] = (
        df_all["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    )
    df_all["Is_Married"] = 0
    df_all.loc[df_all["Title"] == "Mrs", "Is_Married"] = 1
    df_all["Title"] = df_all["Title"].replace(
        ["Miss", "Mrs", "Ms", "Mlle", "Lady", "Mme", "the Countess", "Dona"],
        "Miss/Mrs/Ms",
    )
    df_all["Title"] = df_all["Title"].replace(
        ["Dr", "Col", "Major", "Jonkheer", "Capt", "Sir", "Don", "Rev"],
        "Dr/Military/Noble/Clergy",
    )

    df_all["Family"] = extract_surname(df_all["Name"])


def _target_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    non_unique_families = [
        x for x in df_train["Family"].unique() if x in df_test["Family"].unique()
    ]
    non_unique_tickets = [
        x for x in df_train["Ticket"].unique() if x in df_test["Ticket"].unique()
    ]

    # Old pandas median() skipped non-numeric columns; iloc[i, 1] is Family_Size / Ticket_Frequency.
    df_family_survival_rate = df_train.groupby("Family")[
        ["Survived", "Family_Size"]
    ].median()
    df_ticket_survival_rate = df_train.groupby("Ticket")[
        ["Survived", "Ticket_Frequency"]
    ].median()

    family_rates: dict[str, float] = {}
    ticket_rates: dict[str, float] = {}

    for i in range(len(df_family_survival_rate)):
        if (
            df_family_survival_rate.index[i] in non_unique_families
            and df_family_survival_rate.iloc[i, 1] > 1
        ):
            family_rates[df_family_survival_rate.index[i]] = float(
                df_family_survival_rate.iloc[i, 0]
            )

    for i in range(len(df_ticket_survival_rate)):
        if (
            df_ticket_survival_rate.index[i] in non_unique_tickets
            and df_ticket_survival_rate.iloc[i, 1] > 1
        ):
            ticket_rates[df_ticket_survival_rate.index[i]] = float(
                df_ticket_survival_rate.iloc[i, 0]
            )

    mean_survival_rate = float(np.mean(df_train["Survived"]))

    train_family_survival_rate: list[float] = []
    train_family_survival_rate_na: list[int] = []
    test_family_survival_rate: list[float] = []
    test_family_survival_rate_na: list[int] = []

    for i in range(len(df_train)):
        family = df_train["Family"].iloc[i]
        if family in family_rates:
            train_family_survival_rate.append(family_rates[family])
            train_family_survival_rate_na.append(1)
        else:
            train_family_survival_rate.append(mean_survival_rate)
            train_family_survival_rate_na.append(0)

    for i in range(len(df_test)):
        family = df_test["Family"].iloc[i]
        if family in family_rates:
            test_family_survival_rate.append(family_rates[family])
            test_family_survival_rate_na.append(1)
        else:
            test_family_survival_rate.append(mean_survival_rate)
            test_family_survival_rate_na.append(0)

    df_train["Family_Survival_Rate"] = train_family_survival_rate
    df_train["Family_Survival_Rate_NA"] = train_family_survival_rate_na
    df_test["Family_Survival_Rate"] = test_family_survival_rate
    df_test["Family_Survival_Rate_NA"] = test_family_survival_rate_na

    train_ticket_survival_rate: list[float] = []
    train_ticket_survival_rate_na: list[int] = []
    test_ticket_survival_rate: list[float] = []
    test_ticket_survival_rate_na: list[int] = []

    for i in range(len(df_train)):
        ticket = df_train["Ticket"].iloc[i]
        if ticket in ticket_rates:
            train_ticket_survival_rate.append(ticket_rates[ticket])
            train_ticket_survival_rate_na.append(1)
        else:
            train_ticket_survival_rate.append(mean_survival_rate)
            train_ticket_survival_rate_na.append(0)

    for i in range(len(df_test)):
        ticket = df_test["Ticket"].iloc[i]
        if ticket in ticket_rates:
            test_ticket_survival_rate.append(ticket_rates[ticket])
            test_ticket_survival_rate_na.append(1)
        else:
            test_ticket_survival_rate.append(mean_survival_rate)
            test_ticket_survival_rate_na.append(0)

    df_train["Ticket_Survival_Rate"] = train_ticket_survival_rate
    df_train["Ticket_Survival_Rate_NA"] = train_ticket_survival_rate_na
    df_test["Ticket_Survival_Rate"] = test_ticket_survival_rate
    df_test["Ticket_Survival_Rate_NA"] = test_ticket_survival_rate_na

    for frame in (df_train, df_test):
        frame["Survival_Rate"] = (
            frame["Ticket_Survival_Rate"] + frame["Family_Survival_Rate"]
        ) / 2
        frame["Survival_Rate_NA"] = (
            frame["Ticket_Survival_Rate_NA"] + frame["Family_Survival_Rate_NA"]
        ) / 2


def _encode_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs = [df_train, df_test]
    for df in dfs:
        for feature in NON_NUMERIC_FEATURES:
            df[feature] = LabelEncoder().fit_transform(df[feature])

    encoded_features: list[pd.DataFrame] = []
    for df in dfs:
        for feature in CAT_FEATURES:
            encoded_feat = OneHotEncoder().fit_transform(
                df[feature].values.reshape(-1, 1)
            ).toarray()
            n_unique = df[feature].nunique()
            cols = [f"{feature}_{k}" for k in range(1, n_unique + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = df.index
            encoded_features.append(encoded_df)

    # notebook cell 39
    train_out = pd.concat([df_train, *encoded_features[:6]], axis=1)
    test_out = pd.concat([df_test, *encoded_features[6:]], axis=1)
    return train_out, test_out


def build_geeky837b_matrices(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, list[str]]:
    """Notebook-faithful feature matrices (full train + test, pre-CV)."""
    df_all = concat_df(df_train_raw, df_test_raw)
    _impute_and_deck(df_all)
    _feature_engineering(df_all)

    df_train = df_all.loc[:TRAIN_END].copy()
    df_test = df_all.loc[TRAIN_END + 1 :].copy()
    test_passenger_ids = df_test["PassengerId"].copy()

    _target_encoding(df_train, df_test)
    df_train, df_test = _encode_features(df_train, df_test)

    train_feature_df = df_train.drop(columns=DROP_COLS)
    test_feature_df = df_test.drop(columns=DROP_COLS)

    x_train = StandardScaler().fit_transform(train_feature_df)
    y_train = df_train["Survived"].values
    x_test = StandardScaler().fit_transform(test_feature_df)

    return x_train, y_train, x_test, test_passenger_ids, list(train_feature_df.columns)


def cross_validate_leaderboard(x_train: np.ndarray, y_train: np.ndarray) -> list[float]:
    skf = StratifiedKFold(n_splits=N_FOLDS, random_state=N_FOLDS, shuffle=True)
    scores: list[float] = []
    for trn_idx, val_idx in skf.split(x_train, y_train):
        model = RandomForestClassifier(**LEADERBOARD_RF_PARAMS)
        model.fit(x_train[trn_idx], y_train[trn_idx])
        scores.append(float(accuracy_score(y_train[val_idx], model.predict(x_train[val_idx]))))
    return scores


def predict_leaderboard_submission(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    """5-fold test prob average — notebook submission recipe."""
    skf = StratifiedKFold(n_splits=N_FOLDS, random_state=N_FOLDS, shuffle=True)
    prob_columns = [
        f"Fold_{fold}_Prob_{label}"
        for fold in range(1, N_FOLDS + 1)
        for label in (0, 1)
    ]
    probs = pd.DataFrame(np.zeros((len(x_test), len(prob_columns))), columns=prob_columns)

    for fold, (trn_idx, _) in enumerate(skf.split(x_train, y_train), 1):
        model = RandomForestClassifier(**LEADERBOARD_RF_PARAMS)
        model.fit(x_train[trn_idx], y_train[trn_idx])
        test_proba = model.predict_proba(x_test)
        probs[f"Fold_{fold}_Prob_0"] = test_proba[:, 0]
        probs[f"Fold_{fold}_Prob_1"] = test_proba[:, 1]

    class_survived = [col for col in probs.columns if col.endswith("Prob_1")]
    probs["1"] = probs[class_survived].sum(axis=1) / N_FOLDS
    probs["pred"] = 0
    probs.loc[probs["1"] >= 0.5, "pred"] = 1
    return probs["pred"].astype(int).values
