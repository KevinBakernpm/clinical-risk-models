"""30-day emergency readmission risk and discharge follow-up allocation.

Predicts, at the point of discharge, the probability that an adult inpatient
will be readmitted as an emergency within 30 days. A random forest is
trained on demographics (age, sex, ethnicity, deprivation decile), number
of comorbidities, admissions in the previous 12 months, admission labs,
presenting complaint and length of stay.

Because the transitional-care team has limited capacity, discharged
patients are ranked by predicted risk each week: the highest-risk share is
allocated a community nursing home visit, the next share a telephone
follow-up within 72 hours, and the remainder receive standard discharge
information only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, ReadmissionConfig
from .features import tabular_preprocessor

FOLLOW_UP_TIERS = ("community_visit", "nurse_call", "standard")


@dataclass
class ReadmissionMetrics:
    auroc: float
    brier: float
    readmissions_captured: float
    """Share of actual readmissions that were allocated an active follow-up."""


class ReadmissionRiskModel:
    """Random forest that ranks discharged inpatients for post-discharge follow-up."""

    def __init__(self, config: ReadmissionConfig | None = None) -> None:
        self.config = config or ReadmissionConfig()
        self.pipeline = Pipeline([
            ("prep", tabular_preprocessor(self.config.numeric, self.config.categorical)),
            ("forest", RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                min_samples_leaf=self.config.min_samples_leaf,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ])

    @property
    def features(self) -> list[str]:
        return self.config.numeric + self.config.categorical

    def fit(self, discharges: pd.DataFrame, readmitted: pd.Series) -> "ReadmissionRiskModel":
        self.pipeline.fit(discharges[self.features], readmitted)
        return self

    def predict_proba(self, discharges: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(discharges[self.features])[:, 1]

    def allocate_follow_up(self, discharges: pd.DataFrame) -> pd.DataFrame:
        """Assign each discharge a follow-up tier according to risk rank and team capacity."""
        risk = self.predict_proba(discharges)
        n = len(risk)
        order = np.argsort(-risk, kind="stable")
        n_visit = int(round(n * self.config.community_visit_capacity))
        n_call = int(round(n * self.config.nurse_call_capacity))
        tier = np.full(n, FOLLOW_UP_TIERS[2], dtype=object)
        tier[order[:n_visit]] = FOLLOW_UP_TIERS[0]
        tier[order[n_visit:n_visit + n_call]] = FOLLOW_UP_TIERS[1]
        return pd.DataFrame({
            "encounter_id": discharges["encounter_id"].to_numpy(),
            "readmission_risk": risk.round(3),
            "follow_up": tier,
        }).sort_values("readmission_risk", ascending=False).reset_index(drop=True)

    def feature_importance(self, n: int = 8) -> pd.Series:
        names = self.pipeline.named_steps["prep"].get_feature_names_out()
        importances = self.pipeline.named_steps["forest"].feature_importances_
        return pd.Series(importances, index=names).sort_values(ascending=False).head(n)

    def evaluate(self, discharges: pd.DataFrame, readmitted: pd.Series) -> ReadmissionMetrics:
        proba = self.predict_proba(discharges)
        truth = readmitted.to_numpy()
        allocation = self.allocate_follow_up(discharges).set_index("encounter_id")
        active = allocation.loc[discharges["encounter_id"], "follow_up"].to_numpy() != "standard"
        captured = float(active[truth == 1].mean()) if (truth == 1).any() else float("nan")
        return ReadmissionMetrics(
            auroc=float(roc_auc_score(truth, proba)),
            brier=float(brier_score_loss(truth, proba)),
            readmissions_captured=captured,
        )
