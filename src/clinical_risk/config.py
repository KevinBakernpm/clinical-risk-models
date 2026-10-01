"""Central configuration: feature sets, model hyperparameters and operating thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field

RANDOM_STATE = 42
TEST_SIZE = 0.25

DEMOGRAPHIC_NUMERIC = ["age", "imd_decile"]
DEMOGRAPHIC_CATEGORICAL = ["sex", "ethnicity"]

TRIAGE_VITALS = [
    "heart_rate", "resp_rate", "systolic_bp", "diastolic_bp",
    "temperature", "spo2", "gcs", "pain_score",
]
FIRST_LABS = ["lactate", "wbc", "creatinine", "crp"]
HISTORY = ["comorbidity_count", "prior_admissions_12m"]

WARD_OBS_SIGNALS = ["heart_rate", "resp_rate", "systolic_bp", "temperature", "spo2", "lactate"]


@dataclass(frozen=True)
class TriageConfig:
    """Gradient boosting settings for ED acuity prediction (acuity 1 = most urgent)."""

    numeric: list[str] = field(
        default_factory=lambda: DEMOGRAPHIC_NUMERIC + TRIAGE_VITALS + HISTORY
    )
    categorical: list[str] = field(
        default_factory=lambda: DEMOGRAPHIC_CATEGORICAL + ["arrival_mode", "presenting_complaint"]
    )
    n_estimators: int = 150
    learning_rate: float = 0.08
    max_depth: int = 3


@dataclass(frozen=True)
class DeteriorationConfig:
    """Logistic early-warning score over a rolling window of ward observations."""

    window_hours: int = 12
    numeric_static: list[str] = field(default_factory=lambda: ["age", "comorbidity_count"])
    categorical_static: list[str] = field(default_factory=lambda: ["sex"])
    C: float = 0.5
    # Score bands on a 0-100 scale used to drive the escalation protocol.
    medium_threshold: float = 30.0
    high_threshold: float = 60.0


@dataclass(frozen=True)
class ReadmissionConfig:
    """Random forest settings and follow-up allocation policy for discharged patients."""

    numeric: list[str] = field(
        default_factory=lambda: DEMOGRAPHIC_NUMERIC + HISTORY + FIRST_LABS + ["length_of_stay_days"]
    )
    categorical: list[str] = field(
        default_factory=lambda: DEMOGRAPHIC_CATEGORICAL + ["presenting_complaint"]
    )
    n_estimators: int = 300
    min_samples_leaf: int = 5
    # Share of discharges the transitional-care team can contact each week.
    community_visit_capacity: float = 0.10
    nurse_call_capacity: float = 0.20


@dataclass(frozen=True)
class NoteClassifierConfig:
    """TF-IDF + linear SVM settings for coding free-text triage notes."""

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    C: float = 0.8
    review_margin: float = 0.25


@dataclass(frozen=True)
class LengthOfStayConfig:
    """Regressor settings for predicted length of stay and bed demand."""

    numeric: list[str] = field(
        default_factory=lambda: DEMOGRAPHIC_NUMERIC + TRIAGE_VITALS + FIRST_LABS + HISTORY
    )
    categorical: list[str] = field(
        default_factory=lambda: DEMOGRAPHIC_CATEGORICAL + ["arrival_mode", "presenting_complaint"]
    )
    max_iter: int = 200
    learning_rate: float = 0.06
    forecast_days: int = 7
