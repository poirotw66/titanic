"""Feature engineering from Gunes Evitan / Geeky Codes 0.837 tutorial."""

from __future__ import annotations

import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

FAMILY_SIZE_MAP = {
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

TITLE_FEMALE = ["Miss", "Mrs", "Ms", "Mlle", "Lady", "Mme", "the Countess", "Dona"]
TITLE_OFFICIAL = ["Dr", "Col", "Major", "Jonkheer", "Capt", "Sir", "Don", "Rev"]

CAT_FEATURES = ["Pclass", "Sex", "Deck", "Embarked", "Title", "Family_Size_Grouped"]
LABEL_ENCODE_FEATURES = [
    "Embarked",
    "Sex",
    "Deck",
    "Title",
    "Family_Size_Grouped",
    "Age",
    "Fare",
]
DROP_COLS = [
    "Deck",
    "Embarked",
    "Family",
    "Family_Size",
    "Family_Size_Grouped",
    "Name",
    "Parch",
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


def engineer_geeky837(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return scaled feature matrices and column names."""
    train_size = len(train_features)
    df = pd.concat(
        [train_features.reset_index(drop=True), test_features.reset_index(drop=True)],
        ignore_index=True,
    )
    y_train = y_train.reset_index(drop=True)

    _impute_and_deck(df)
    _bin_continuous(df)
    _add_frequency_features(df)
    _add_title_features(df)
    df["Family"] = _extract_surnames(df["Name"])

    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    train_df["Survived"] = y_train.values

    _add_target_encoding(train_df, test_df)
    _encode_features(train_df, test_df)

    train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns], inplace=True)
    test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns], inplace=True)

    feature_names = list(train_df.columns)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df)
    x_test = scaler.transform(test_df)
    return x_train, x_test, feature_names


def _impute_and_deck(df: pd.DataFrame) -> None:
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda series: series.fillna(series.median())
    )
    df["Embarked"] = df["Embarked"].fillna("S")

    fare_median = df.groupby(["Pclass", "Parch", "SibSp"])["Fare"].transform("median")
    df["Fare"] = df["Fare"].fillna(fare_median)

    df["Deck"] = df["Cabin"].apply(lambda value: value[0] if pd.notnull(value) else "M")
    df.loc[df["Deck"] == "T", "Deck"] = "A"
    df["Deck"] = df["Deck"].replace(["A", "B", "C"], "ABC")
    df["Deck"] = df["Deck"].replace(["D", "E"], "DE")
    df["Deck"] = df["Deck"].replace(["F", "G"], "FG")
    df.drop(columns=["Cabin"], inplace=True)


def _bin_continuous(df: pd.DataFrame) -> None:
    df["Fare"] = pd.qcut(df["Fare"], 13, duplicates="drop")
    df["Age"] = pd.qcut(df["Age"], 10, duplicates="drop")


def _add_frequency_features(df: pd.DataFrame) -> None:
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df["Family_Size_Grouped"] = df["Family_Size"].map(FAMILY_SIZE_MAP)
    df["Ticket_Frequency"] = df.groupby("Ticket")["Ticket"].transform("count")


def _add_title_features(df: pd.DataFrame) -> None:
    df["Title"] = df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    df["Is_Married"] = (df["Title"] == "Mrs").astype(int)
    df["Title"] = df["Title"].replace(TITLE_FEMALE, "Miss/Mrs/Ms")
    df["Title"] = df["Title"].replace(TITLE_OFFICIAL, "Dr/Military/Noble/Clergy")


def _extract_surnames(names: pd.Series) -> list[str]:
    families: list[str] = []
    for name in names:
        name_no_paren = name.split("(")[0] if "(" in name else name
        family = name_no_paren.split(",")[0]
        for char in string.punctuation:
            family = family.replace(char, "").strip()
        families.append(family)
    return families


def _add_target_encoding(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    non_unique_families = set(train_df["Family"]) & set(test_df["Family"])
    non_unique_tickets = set(train_df["Ticket"]) & set(test_df["Ticket"])
    mean_rate = float(train_df["Survived"].mean())

    family_stats = train_df.groupby("Family").agg(
        Survived=("Survived", "median"),
        Family_Size=("Family_Size", "median"),
    )
    ticket_stats = train_df.groupby("Ticket").agg(
        Survived=("Survived", "median"),
        Ticket_Frequency=("Ticket_Frequency", "median"),
    )

    family_rates = {
        name: float(row["Survived"])
        for name, row in family_stats.iterrows()
        if name in non_unique_families and row["Family_Size"] > 1
    }
    ticket_rates = {
        ticket: float(row["Survived"])
        for ticket, row in ticket_stats.iterrows()
        if ticket in non_unique_tickets and row["Ticket_Frequency"] > 1
    }

    for frame in (train_df, test_df):
        family_rate, family_na = _map_survival_rate(
            frame["Family"], family_rates, mean_rate
        )
        ticket_rate, ticket_na = _map_survival_rate(
            frame["Ticket"], ticket_rates, mean_rate
        )
        frame["Family_Survival_Rate"] = family_rate
        frame["Family_Survival_Rate_NA"] = family_na
        frame["Ticket_Survival_Rate"] = ticket_rate
        frame["Ticket_Survival_Rate_NA"] = ticket_na
        frame["Survival_Rate"] = (frame["Ticket_Survival_Rate"] + frame["Family_Survival_Rate"]) / 2
        frame["Survival_Rate_NA"] = (
            frame["Ticket_Survival_Rate_NA"] + frame["Family_Survival_Rate_NA"]
        ) / 2


def _map_survival_rate(
    keys: pd.Series,
    rate_map: dict[object, float],
    default_rate: float,
) -> tuple[list[float], list[int]]:
    rates: list[float] = []
    flags: list[int] = []
    for key in keys:
        if key in rate_map:
            rates.append(rate_map[key])
            flags.append(1)
        else:
            rates.append(default_rate)
            flags.append(0)
    return rates, flags


def _encode_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    combined = pd.concat([train_df, test_df], axis=0)
    for feature in LABEL_ENCODE_FEATURES:
        encoder = LabelEncoder()
        encoder.fit(combined[feature].astype(str))
        train_df[feature] = encoder.transform(train_df[feature].astype(str))
        test_df[feature] = encoder.transform(test_df[feature].astype(str))

    for feature in CAT_FEATURES:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(combined[[feature]].astype(str))
        train_encoded = pd.DataFrame(
            encoder.transform(train_df[[feature]].astype(str)),
            columns=[f"{feature}_{i}" for i in range(encoder.categories_[0].shape[0])],
            index=train_df.index,
        )
        test_encoded = pd.DataFrame(
            encoder.transform(test_df[[feature]].astype(str)),
            columns=train_encoded.columns,
            index=test_df.index,
        )
        train_df.drop(columns=[feature], inplace=True)
        test_df.drop(columns=[feature], inplace=True)
        for column in train_encoded.columns:
            train_df[column] = train_encoded[column].values
            test_df[column] = test_encoded[column].values

    if "Survived" in train_df.columns:
        train_df.drop(columns=["Survived"], inplace=True)
