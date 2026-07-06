"""Tier 1 feature engineering for Titanic survival prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RARE_TITLES = frozenset(
    {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
)
TITLE_REPLACEMENTS = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}

TIER1_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Title",
    "Embarked",
]
TIER2_COLUMNS = ["FarePerPerson", "Deck"]
CAT_FEATURE_NAMES = ["Pclass", "Sex", "Title", "Embarked"]


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Extract Title, family features, and group-wise imputation (fit on train fold only)."""

    def __init__(self, tier: int = 1) -> None:
        self.tier = tier

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> TitanicFeatureEngineer:
        df = _add_derived_columns(X.copy())
        embarked = df["Embarked"].dropna()
        self.embarked_fill_ = (
            embarked.mode().iloc[0] if len(embarked) else "S"
        )

        filled = df.copy()
        filled["Embarked"] = filled["Embarked"].fillna(self.embarked_fill_)

        age_known = filled.dropna(subset=["Age"])
        self.age_median_ = age_known.groupby(["Title", "Pclass"], observed=True)[
            "Age"
        ].median()
        self.global_age_median_ = (
            float(age_known["Age"].median()) if len(age_known) else 28.0
        )

        self.fare_median_by_pclass_ = filled.groupby("Pclass", observed=True)[
            "Fare"
        ].median()
        self.global_fare_median_ = float(filled["Fare"].median())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = _add_derived_columns(X.copy())
        df["Embarked"] = df["Embarked"].fillna(self.embarked_fill_)
        df["Age"] = _impute_age(df, self.age_median_, self.global_age_median_)
        df["Fare"] = _impute_fare(
            df, self.fare_median_by_pclass_, self.global_fare_median_
        )

        columns = list(TIER1_COLUMNS)
        if self.tier >= 2:
            df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
            df["Deck"] = df["Cabin"].str[0].fillna("U")
            columns.extend(TIER2_COLUMNS)

        return df[columns]

    def cat_feature_indices(self) -> list[int]:
        columns = list(TIER1_COLUMNS)
        if self.tier >= 2:
            columns.extend(TIER2_COLUMNS)
        return [columns.index(name) for name in CAT_FEATURE_NAMES]


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(TITLE_REPLACEMENTS)
    df.loc[df["Title"].isin(RARE_TITLES), "Title"] = "Rare"
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def _impute_age(
    df: pd.DataFrame,
    age_median: pd.Series,
    global_median: float,
) -> pd.Series:
    result = df["Age"].copy()
    missing = result.isna()
    if not missing.any():
        return result

    lookup = (
        df.loc[missing, ["Title", "Pclass"]]
        .merge(
            age_median.reset_index(name="_age"),
            on=["Title", "Pclass"],
            how="left",
        )["_age"]
    )
    result.loc[missing] = lookup.values
    return result.fillna(global_median)


def _impute_fare(
    df: pd.DataFrame,
    fare_median_by_pclass: pd.Series,
    global_median: float,
) -> pd.Series:
    result = df["Fare"].copy()
    missing = result.isna()
    if not missing.any():
        return result

    result.loc[missing] = df.loc[missing, "Pclass"].map(fare_median_by_pclass)
    return result.fillna(global_median)
