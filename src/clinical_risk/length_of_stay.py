"""Length-of-stay prediction and inpatient bed demand forecasting.

Predicts the expected length of stay in days for each patient admitted from
the emergency department, using a histogram gradient boosting regressor on
demographics (age, sex, ethnicity, deprivation decile), arrival vital signs,
admission labs, presenting complaint and medical history.

Predicted stays are rolled forward into a day-by-day forecast of occupied
beds, which the site operations team uses at the morning bed meeting to
plan elective cancellations, escalation beds and discharge lounge staffing.
Individual predictions also set the expected discharge date shown on the
ward board.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, LengthOfStayConfig
from .features import tabular_preprocessor


@dataclass
class LengthOfStayMetrics:
    mae_days: float
    r2_log: float
    within_one_day: float


class LengthOfStayModel:
    """Gradient boosting regressor for length of stay, trained on log(1 + days)."""

    def __init__(self, config: LengthOfStayConfig | None = None) -> None:
        self.config = config or LengthOfStayConfig()
        self.pipeline = Pipeline([
            ("prep", tabular_preprocessor(self.config.numeric, self.config.categorical)),
            ("hgb", HistGradientBoostingRegressor(
                max_iter=self.config.max_iter,
                learning_rate=self.config.learning_rate,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            )),
        ])

    @property
    def features(self) -> list[str]:
        return self.config.numeric + self.config.categorical

    def fit(self, admissions: pd.DataFrame, los_days: pd.Series) -> "LengthOfStayModel":
        self.pipeline.fit(admissions[self.features], np.log1p(los_days))
        return self

    def predict(self, admissions: pd.DataFrame) -> np.ndarray:
        """Predicted length of stay in days (never negative)."""
        return np.clip(np.expm1(self.pipeline.predict(admissions[self.features])), 0.0, None)

    def expected_discharge(self, admissions: pd.DataFrame) -> pd.DataFrame:
        los = self.predict(admissions)
        arrival = pd.to_datetime(admissions["arrival_time"]).to_numpy()
        return pd.DataFrame({
            "encounter_id": admissions["encounter_id"].to_numpy(),
            "predicted_los_days": los.round(1),
            "expected_discharge": pd.to_datetime(arrival) + pd.to_timedelta(los, unit="D"),
        })

    def forecast_bed_demand(self, admissions: pd.DataFrame, start: pd.Timestamp,
                            days: int | None = None) -> pd.DataFrame:
        """Expected occupied beds at midnight for each of the next ``days`` days.

        Counts every admission whose arrival precedes the census point and
        whose predicted discharge falls after it.
        """
        days = days or self.config.forecast_days
        stays = self.expected_discharge(admissions)
        arrival = pd.to_datetime(admissions["arrival_time"]).to_numpy()
        discharge = stays["expected_discharge"].to_numpy()
        census = pd.date_range(start.normalize() + pd.Timedelta(days=1), periods=days, freq="D")
        occupied = [int(((arrival <= t) & (discharge > t)).sum()) for t in census.to_numpy()]
        return pd.DataFrame({"census_date": census.date, "expected_occupied_beds": occupied})

    def evaluate(self, admissions: pd.DataFrame, los_days: pd.Series) -> LengthOfStayMetrics:
        pred = self.predict(admissions)
        truth = los_days.to_numpy()
        return LengthOfStayMetrics(
            mae_days=float(mean_absolute_error(truth, pred)),
            r2_log=float(r2_score(np.log1p(truth), np.log1p(pred))),
            within_one_day=float(np.mean(np.abs(truth - pred) <= 1.0)),
        )
