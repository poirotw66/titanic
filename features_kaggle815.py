"""Feature engineering ported from Kaggle 815 notebook (eu1234)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

DECK_LEVEL = {"G": 1, "F": 2, "E": 3, "D": 4, "C": 5, "B": 6, "A": 7}

PCLASS_DECK_PRICE_RANGES: dict[int, dict[str, list[float]]] = {
    1: {"A": [25, 30], "B": [35, 70], "C": [30, 35], "D": [19, 25], "E": [9, 19]},
    2: {"D": [13, 17], "E": [5, 9], "F": [9, 13]},
    3: {"E": [8, 9], "F": [9, 21], "G": [0, 8]},
}


def engineer_kaggle815(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build model matrices using train+test combined FE (notebook recipe)."""
    train_size = len(train_features)
    df = pd.concat(
        [train_features.reset_index(drop=True), test_features.reset_index(drop=True)],
        axis=0,
        ignore_index=True,
    )
    y_train = y_train.reset_index(drop=True)

    df.loc[df["Fare"].eq(0), "Fare"] = np.nan
    df["Lastname"] = df["Name"].str.split(", ").str[0]
    _add_title_and_status(df)
    _add_ticket_and_price(df)
    _add_deck_features(df, train_features, y_train)
    _add_family_features(df, train_size, y_train)
    _impute_and_encode(df)
    _add_binned_features(df)
    _scale_features(df)

    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()


def _add_title_and_status(df: pd.DataFrame) -> None:
    df["Title"] = df["Name"].str.split(", ").str[1].str.split(".").str[0]

    df.loc[df["Title"].isin(["Mrs", "Mme"]), "Title"] = "Mrs"
    df.loc[df["Title"].isin(["Ms", "Miss", "Mlle"]), "Title"] = "Miss"
    df.loc[(df["Title"].eq("Mr") & df["Age"].le(18)), "Title"] = "Master"
    df.loc[(df["Title"].eq("Mrs") & df["Age"].le(18)), "Title"] = "Miss"
    df.loc[
        (~df["Title"].isin(["Mrs", "Miss", "Mr", "Master"]) & df["Sex"].eq("male")),
        "Title",
    ] = "Mr"
    df.loc[
        (~df["Title"].isin(["Mrs", "Miss", "Mr", "Master"]) & df["Sex"].eq("female")),
        "Title",
    ] = "Mrs"

    df.loc[df["Age"].le(18), "Status"] = "Child"
    df.loc[df["Title"].eq("Master"), "Status"] = "Child"
    df.loc[(df["Title"].eq("Mrs") & df["Parch"].gt(0)), "Status"] = "Mother"
    df.loc[(df["Status"].isna() & df["Sex"].eq("female")), "Status"] = "Woman"
    df.loc[(df["Status"].isna() & df["Sex"].eq("male")), "Status"] = "Man"


def _add_ticket_and_price(df: pd.DataFrame) -> None:
    split_ticket = df["Ticket"].astype(str).str.split()
    df["Ticket_series"] = split_ticket.apply(lambda parts: parts[0] if len(parts) > 1 else 0)
    df["Ticket_nr"] = split_ticket.apply(lambda parts: parts[-1])
    ticket_counts = df.groupby("Ticket_nr")["Lastname"].count().to_dict()
    df["Passengers_ticket"] = df["Ticket_nr"].map(ticket_counts)
    df["Price"] = (df["Fare"] / df["Passengers_ticket"]).round(1)


def _impute_deck_by(df: pd.DataFrame, feature: str) -> None:
    for pclass in range(1, 4):
        mapping = (
            df.loc[~df["Deck"].isna() & df["Pclass"].eq(pclass)]
            .groupby(feature)["Deck"]
            .unique()
            .apply(list)
            .to_dict()
        )
        mapping = {key: values[0] for key, values in mapping.items() if len(values) == 1}
        mask = df["Deck"].isna() & df["Pclass"].eq(pclass)
        df.loc[mask, "Deck"] = df.loc[mask, feature].map(mapping)


def _deck_survival_ratios(train_features: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    train = train_features.copy()
    train["Survived"] = y_train.values
    train["Deck"] = train["Cabin"].str[0]
    survived = train.groupby("Deck")["Survived"].sum()
    counts = train.groupby("Deck")["Deck"].count()
    ratios = (survived / counts).round(2).to_dict()
    missing = train.loc[train["Deck"].isna(), "Survived"]
    ratios["M"] = round(float(missing.mean()), 2) if len(missing) else 0.0
    return ratios


def _add_deck_features(
    df: pd.DataFrame,
    train_features: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    df["Deck"] = df["Cabin"].str[0]
    _impute_deck_by(df, "Ticket_nr")
    _impute_deck_by(df, "Lastname")

    ratios = _deck_survival_ratios(train_features, y_train)
    df["Deck_survive_ratio"] = df["Deck"].fillna("M").map(ratios).astype(float)

    df.loc[df["Deck"].eq("T"), "Deck"] = "A"
    df.loc[(df["Deck"].eq("B") & df["Price"].lt(19)), "Price"] = 19
    df.loc[(df["Deck"].eq("B") & df["Price"].gt(68)), "Price"] = 68

    class_deck_price = (
        df.groupby(["Pclass", "Deck"])["Price"].mean().round(2).reset_index()
    )
    for index, row in df.loc[df["Price"].isna(), ["Pclass", "Deck"]].iterrows():
        if not pd.isna(row["Deck"]):
            price = class_deck_price.loc[
                (class_deck_price["Pclass"].eq(row["Pclass"]))
                & (class_deck_price["Deck"].eq(row["Deck"])),
                "Price",
            ].mean()
        else:
            price = class_deck_price.loc[
                class_deck_price["Pclass"].eq(row["Pclass"]), "Price"
            ].mean()
        df.loc[index, "Price"] = price

    for index, row in df.loc[df["Deck"].isna(), ["Pclass", "Price"]].iterrows():
        for pclass, deck_ranges in PCLASS_DECK_PRICE_RANGES.items():
            if row["Pclass"] != pclass:
                continue
            for deck, bounds in deck_ranges.items():
                if bounds[1] > row["Price"] >= bounds[0]:
                    df.loc[index, "Deck"] = deck
                    break

    df["Deck"] = df["Deck"].replace(DECK_LEVEL)

    deck_people = df["Deck"].value_counts().sort_index().to_dict()
    escape_density: dict[int, int] = {}
    remaining = dict(deck_people)
    for level in range(1, 8):
        escape_density[level] = sum(remaining.values())
        remaining.pop(level, None)
    df["Escape_density"] = df["Deck"].replace(escape_density)


def _add_family_features(
    df: pd.DataFrame, train_size: int, y_train: pd.Series
) -> None:
    df["Family_size"] = 1 + df["SibSp"] + df["Parch"]

    train_slice = df.iloc[:train_size].copy()
    train_slice["Survived"] = y_train.values
    family_rates = (
        train_slice.groupby("Lastname")["Survived"].mean().round(2).to_dict()
    )
    test_lastnames = set(df.iloc[train_size:]["Lastname"].unique())
    common = {
        name: rate
        for name, rate in family_rates.items()
        if name in test_lastnames
    }
    df["Family_survivers"] = df["Lastname"].map(common)
    df["Family_survivers"] = df["Family_survivers"].fillna(df["Family_survivers"].mean())

    df["Pclass"] = df["Pclass"].astype("object")
    drop_cols = [
        "Name",
        "Ticket",
        "Ticket_nr",
        "Ticket_series",
        "Fare",
        "Cabin",
        "Lastname",
        "Passengers_ticket",
    ]
    df.drop(columns=drop_cols, inplace=True)


def _impute_and_encode(df: pd.DataFrame) -> None:
    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    cat_imputer = SimpleImputer(strategy="most_frequent")
    encoded = pd.DataFrame(
        cat_imputer.fit_transform(df[categorical_cols]),
        columns=categorical_cols,
    )
    encoded = pd.get_dummies(encoded)

    num_imputer = IterativeImputer(random_state=42)
    numeric = pd.DataFrame(
        num_imputer.fit_transform(df[numeric_cols]),
        columns=numeric_cols,
    )

    combined = pd.concat([encoded, numeric], axis=1)
    df.drop(df.columns, axis=1, inplace=True)
    for column in combined.columns:
        df[column] = combined[column].values


def _add_binned_features(df: pd.DataFrame) -> None:
    df["Age_group"] = pd.cut(
        x=df["Age"],
        labels=[5, 1, 4, 3, 2],
        bins=[-1, 15, 33, 45, 60, df["Age"].max()],
    ).astype(float)
    df["Family_group"] = pd.cut(
        x=df["Family_size"],
        labels=[1, 3, 2],
        bins=[-1, 1, 4, df["Family_size"].max()],
    ).astype(float)
    df["Lucky_family"] = pd.cut(
        x=df["Family_survivers"],
        labels=[2, 3, 1, 4],
        bins=[-1, 0.22, 0.35, 0.49, df["Family_survivers"].max()],
    ).astype(float)


def _scale_features(df: pd.DataFrame) -> None:
    df["Price"] = df["Price"].apply(np.log1p)
    columns = list(df.columns)
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df), columns=columns)
    if "Family_survivers" in scaled.columns:
        scaled = scaled.drop(columns=["Family_survivers"])
    df.drop(df.columns, axis=1, inplace=True)
    for column in scaled.columns:
        df[column] = scaled[column].values
