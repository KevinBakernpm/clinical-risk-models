"""Inpatient early warning score for sepsis and clinical deterioration.

Estimates the probability that an admitted adult patient will deteriorate
(unplanned ICU transfer, cardiac arrest call, or sepsis bundle trigger)
within the next 48 hours. A regularised logistic regression is trained on
time-windowed ward observations - latest value, mean, range and hourly
trend of heart rate, respiratory rate, blood pressure, temperature, oxygen
saturation and lactate - combined with age, sex and comorbidity burden.

The probability is rescaled to a 0-100 early warning score. Scores in the
high band page the critical care outreach team for bedside review within
30 minutes; the medium band increases the observation frequency to hourly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, WARD_OBS_SIGNALS, DeteriorationConfig
from .features import tabular_preprocessor, window_features

ESCALATION_ACTIONS = {
    "low": "routine observations every 4 hours",
    "medium": "hourly observations and nurse-in-charge informed",
    "high": "critical care outreach review within 30 minutes",
}


@dataclass
class DeteriorationMetrics:
    auroc: float
    average_precision: float
    high_band_sensitivity: float
    alert_rate: float


def build_feature_table(observations: pd.DataFrame, encounters: pd.DataFrame,
                        window_hours: int) -> pd.DataFrame:
    """Join windowed observation summaries to static patient attributes."""
    windowed = window_features(observations, WARD_OBS_SIGNALS, window_hours)
    static_cols = ["encounter_id", "age", "sex", "comorbidity_count"]
    return windowed.merge(encounters[static_cols], on="encounter_id", how="left")


class DeteriorationScore:
    """Logistic regression early warning score over rolling ward observations."""

    def __init__(self, config: DeteriorationConfig | None = None) -> None:
        self.config = config or DeteriorationConfig()
        self._numeric: list[str] = []
        self.pipeline: Pipeline | None = None

    def _build(self, table: pd.DataFrame) -> Pipeline:
        excluded = {"encounter_id", *self.config.categorical_static}
        self._numeric = [c for c in table.columns if c not in excluded]
        return Pipeline([
            ("prep", tabular_preprocessor(self._numeric, self.config.categorical_static, scale=True)),
            ("logit", LogisticRegression(
                C=self.config.C, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
            )),
        ])

    def fit(self, table: pd.DataFrame, deteriorated: pd.Series) -> "DeteriorationScore":
        self.pipeline = self._build(table)
        self.pipeline.fit(table, deteriorated)
        return self

    def _check_fitted(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("DeteriorationScore must be fitted before scoring")
        return self.pipeline

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        return self._check_fitted().predict_proba(table)[:, 1]

    def score(self, table: pd.DataFrame) -> pd.DataFrame:
        """Return the 0-100 early warning score, alert band and escalation action."""
        points = np.round(self.predict_proba(table) * 100, 1)
        band = np.select(
            [points >= self.config.high_threshold, points >= self.config.medium_threshold],
            ["high", "medium"], default="low",
        )
        return pd.DataFrame({
            "encounter_id": table["encounter_id"].to_numpy(),
            "ews_score": points,
            "band": band,
            "action": [ESCALATION_ACTIONS[b] for b in band],
        })

    def top_drivers(self, n: int = 5) -> pd.Series:
        """Largest standardised coefficients, for clinician-facing explanations."""
        pipe = self._check_fitted()
        names = pipe.named_steps["prep"].get_feature_names_out()
        coefs = pd.Series(pipe.named_steps["logit"].coef_[0], index=names)
        return coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(n)

    def evaluate(self, table: pd.DataFrame, deteriorated: pd.Series) -> DeteriorationMetrics:
        proba = self.predict_proba(table)
        truth = deteriorated.to_numpy()
        high = proba * 100 >= self.config.high_threshold
        positives = truth == 1
        return DeteriorationMetrics(
            auroc=float(roc_auc_score(truth, proba)),
            average_precision=float(average_precision_score(truth, proba)),
            high_band_sensitivity=float(high[positives].mean()) if positives.any() else float("nan"),
            alert_rate=float(high.mean()),
        )
