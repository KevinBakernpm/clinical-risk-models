"""Emergency department triage acuity prediction.

Predicts the five-level triage acuity (1 = resuscitation, 2 = emergency,
3 = urgent, 4 = standard, 5 = non-urgent) for every adult patient arriving
at the emergency department, using a gradient boosted tree ensemble over
arrival vital signs, presenting complaint, arrival mode, medical history and
patient demographics (age, sex, ethnicity, deprivation decile).

The predicted acuity is shown to the triage nurse alongside their own
assessment and determines the patient's initial position in the queue to
be seen by a clinician. Where the model suggests a more urgent category
than the nurse, the case is flagged for senior review.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, TriageConfig
from .features import tabular_preprocessor

ACUITY_LABELS = {1: "resuscitation", 2: "emergency", 3: "urgent", 4: "standard", 5: "non-urgent"}
TARGET_WAIT_MINUTES = {1: 0, 2: 10, 3: 60, 4: 120, 5: 240}


@dataclass
class TriageMetrics:
    macro_f1: float
    weighted_kappa: float
    under_triage_rate: float
    over_triage_rate: float


class TriageAcuityModel:
    """Gradient boosting classifier that assigns a triage acuity level to ED arrivals."""

    def __init__(self, config: TriageConfig | None = None) -> None:
        self.config = config or TriageConfig()
        self.pipeline = Pipeline([
            ("prep", tabular_preprocessor(self.config.numeric, self.config.categorical)),
            ("gbm", GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                random_state=RANDOM_STATE,
            )),
        ])

    @property
    def features(self) -> list[str]:
        return self.config.numeric + self.config.categorical

    def fit(self, encounters: pd.DataFrame, acuity: pd.Series) -> "TriageAcuityModel":
        self.pipeline.fit(encounters[self.features], acuity)
        return self

    def predict(self, encounters: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(encounters[self.features])

    def predict_proba(self, encounters: pd.DataFrame) -> pd.DataFrame:
        proba = self.pipeline.predict_proba(encounters[self.features])
        return pd.DataFrame(proba, columns=self.pipeline.classes_, index=encounters.index)

    def evaluate(self, encounters: pd.DataFrame, acuity: pd.Series) -> TriageMetrics:
        """Agreement with nurse triage; under-triage means predicting a less urgent level."""
        pred = self.predict(encounters)
        truth = acuity.to_numpy()
        return TriageMetrics(
            macro_f1=float(f1_score(truth, pred, average="macro")),
            weighted_kappa=float(cohen_kappa_score(truth, pred, weights="quadratic")),
            under_triage_rate=float(np.mean(pred > truth)),
            over_triage_rate=float(np.mean(pred < truth)),
        )

    def triage_queue(self, encounters: pd.DataFrame,
                     nurse_acuity: pd.Series | None = None) -> pd.DataFrame:
        """Order waiting patients by predicted acuity and flag disagreements for review."""
        proba = self.predict_proba(encounters)
        pred = proba.idxmax(axis=1).astype(int)
        queue = pd.DataFrame({
            "encounter_id": encounters["encounter_id"],
            "predicted_acuity": pred,
            "acuity_label": pred.map(ACUITY_LABELS),
            "confidence": proba.max(axis=1).round(3),
            "target_wait_minutes": pred.map(TARGET_WAIT_MINUTES),
        })
        if nurse_acuity is not None:
            queue["nurse_acuity"] = nurse_acuity.to_numpy()
            queue["senior_review"] = queue["predicted_acuity"] < queue["nurse_acuity"]
        sort_cols = ["predicted_acuity", "confidence"]
        return queue.sort_values(sort_cols, ascending=[True, False]).reset_index(drop=True)
