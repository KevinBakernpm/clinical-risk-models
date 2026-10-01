"""Train, evaluate and demonstrate every model on a cohort of encounters."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from .config import DeteriorationConfig
from .deterioration_score import DeteriorationScore, build_feature_table
from .features import split
from .length_of_stay import LengthOfStayModel
from .note_classifier import TriageNoteClassifier
from .readmission_risk import ReadmissionRiskModel
from .triage_acuity import TriageAcuityModel


def _fmt(metrics: Any) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in asdict(metrics).items())


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def run_triage(encounters: pd.DataFrame) -> dict[str, float]:
    _section("ED triage acuity (gradient boosting)")
    X_train, X_test, y_train, y_test = split(encounters, "acuity")
    model = TriageAcuityModel().fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    print(_fmt(metrics))
    print(model.triage_queue(X_test.head(6), y_test.head(6)).to_string(index=False))
    return asdict(metrics)


def run_deterioration(encounters: pd.DataFrame, observations: pd.DataFrame) -> dict[str, float]:
    _section("Inpatient early warning score (logistic regression)")
    config = DeteriorationConfig()
    table = build_feature_table(observations, encounters, config.window_hours)
    labels = encounters.set_index("encounter_id").loc[table["encounter_id"], "deteriorated_48h"]
    table = table.assign(deteriorated_48h=labels.to_numpy())
    X_train, X_test, y_train, y_test = split(table, "deteriorated_48h")
    model = DeteriorationScore(config).fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    print(_fmt(metrics))
    print("Top drivers:")
    print(model.top_drivers().round(3).to_string())
    print(model.score(X_test.head(5)).to_string(index=False))
    return asdict(metrics)


def run_readmission(encounters: pd.DataFrame) -> dict[str, float]:
    _section("30-day readmission risk (random forest)")
    discharges = encounters[encounters["admitted"] == 1]
    X_train, X_test, y_train, y_test = split(discharges, "readmitted_30d")
    model = ReadmissionRiskModel().fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    print(_fmt(metrics))
    allocation = model.allocate_follow_up(X_test)
    print(allocation["follow_up"].value_counts().to_string())
    print(allocation.head(5).to_string(index=False))
    return asdict(metrics)


def run_note_classifier(encounters: pd.DataFrame) -> dict[str, float]:
    _section("Triage note symptom coding (TF-IDF + linear SVM)")
    X_train, X_test, y_train, y_test = split(encounters[["triage_note", "symptom_code"]], "symptom_code")
    model = TriageNoteClassifier().fit(X_train["triage_note"], y_train)
    metrics = model.evaluate(X_test["triage_note"], y_test)
    print(_fmt(metrics))
    examples = [
        "pt reports crushing chest pain and sweating",
        "found on floor by carer, painful hip",
        "febrile and confused, nursing home resident",
        "took an overdose of tablets, low mood",
    ]
    print(model.code(examples).to_string(index=False))
    return asdict(metrics)


def run_length_of_stay(encounters: pd.DataFrame) -> dict[str, float]:
    _section("Length of stay and bed demand (gradient boosting regressor)")
    admissions = encounters[encounters["admitted"] == 1]
    X_train, X_test, y_train, y_test = split(admissions, "length_of_stay_days", stratify=False)
    model = LengthOfStayModel().fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    print(_fmt(metrics))
    start = pd.to_datetime(admissions["arrival_time"]).max() - pd.Timedelta(days=7)
    recent = admissions[pd.to_datetime(admissions["arrival_time"]) <= start]
    print(model.forecast_bed_demand(recent, start).to_string(index=False))
    return asdict(metrics)


def run_all(encounters: pd.DataFrame, observations: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Run every model end to end and return the held-out metrics for each."""
    print(f"Cohort: {len(encounters)} encounters, {int(encounters['admitted'].sum())} admissions, "
          f"{len(observations)} ward observations")
    return {
        "triage_acuity": run_triage(encounters),
        "deterioration": run_deterioration(encounters, observations),
        "readmission": run_readmission(encounters),
        "note_classifier": run_note_classifier(encounters),
        "length_of_stay": run_length_of_stay(encounters),
    }
