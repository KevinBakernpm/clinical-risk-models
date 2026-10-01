"""Shared feature engineering and preprocessing utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE, TEST_SIZE


def tabular_preprocessor(numeric: Sequence[str], categorical: Sequence[str],
                         scale: bool = False) -> ColumnTransformer:
    """Impute and encode a mixed numeric/categorical table.

    Numeric columns are median-imputed (and optionally standardised);
    categorical columns are one-hot encoded, ignoring unseen levels at
    inference time so new arrival modes or complaints never break scoring.
    """
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), list(numeric)),
        ("cat", categorical_pipe, list(categorical)),
    ])


def split(frame: pd.DataFrame, target: str, stratify: bool = True
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split with a fixed seed, stratified on the target when it is categorical."""
    y = frame[target]
    return train_test_split(
        frame.drop(columns=[target]), y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y if stratify else None,
    )


def shock_index(heart_rate: pd.Series, systolic_bp: pd.Series) -> pd.Series:
    """Heart rate divided by systolic blood pressure; values above ~0.9 suggest shock."""
    return heart_rate / systolic_bp.replace(0, np.nan)


def window_features(observations: pd.DataFrame, signals: Sequence[str],
                    window_hours: int) -> pd.DataFrame:
    """Summarise the most recent ``window_hours`` of observations per encounter.

    For every signal this produces the latest value, the window mean, min,
    max and a least-squares slope (units per hour), giving the model both
    the current state and the direction of travel.
    """
    obs = observations.sort_values(["encounter_id", "hours_since_admission"])
    latest_hour = obs.groupby("encounter_id")["hours_since_admission"].transform("max")
    recent = obs[obs["hours_since_admission"] > latest_hour - window_hours]

    def _slope(group: pd.DataFrame, col: str) -> float:
        hours = group["hours_since_admission"].to_numpy(dtype=float)
        if len(hours) < 2 or np.ptp(hours) == 0:
            return 0.0
        return float(np.polyfit(hours, group[col].to_numpy(dtype=float), 1)[0])

    grouped = recent.groupby("encounter_id")
    parts = []
    for col in signals:
        agg = grouped[col].agg(["last", "mean", "min", "max"])
        agg.columns = [f"{col}_{stat}" for stat in agg.columns]
        agg[f"{col}_slope"] = grouped.apply(lambda g, c=col: _slope(g, c))
        parts.append(agg)
    features = pd.concat(parts, axis=1)
    features["shock_index_last"] = shock_index(
        features["heart_rate_last"], features["systolic_bp_last"]
    )
    return features.reset_index()
