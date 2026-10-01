from __future__ import annotations

import pandas as pd

from clinical_risk.config import DeteriorationConfig
from clinical_risk.deterioration_score import DeteriorationScore, build_feature_table
from clinical_risk.features import split
from clinical_risk.length_of_stay import LengthOfStayModel
from clinical_risk.note_classifier import TriageNoteClassifier, normalise_note
from clinical_risk.readmission_risk import ReadmissionRiskModel
from clinical_risk.triage_acuity import TriageAcuityModel
from generate_synthetic import generate_cohort


def test_synthetic_data_is_deterministic() -> None:
    a, obs_a = generate_cohort(n_encounters=200, seed=11)
    b, obs_b = generate_cohort(n_encounters=200, seed=11)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(obs_a, obs_b)
    assert set(a["acuity"]).issubset({1, 2, 3, 4, 5})
    assert (a.loc[a["admitted"] == 0, "length_of_stay_days"] == 0).all()


def test_triage_model_beats_chance(cohort) -> None:
    encounters, _ = cohort
    X_train, X_test, y_train, y_test = split(encounters, "acuity")
    model = TriageAcuityModel().fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    assert metrics.weighted_kappa > 0.4
    queue = model.triage_queue(X_test.head(10), y_test.head(10))
    assert queue["predicted_acuity"].is_monotonic_increasing


def test_deterioration_score_bands(cohort) -> None:
    encounters, observations = cohort
    config = DeteriorationConfig()
    table = build_feature_table(observations, encounters, config.window_hours)
    labels = encounters.set_index("encounter_id").loc[table["encounter_id"], "deteriorated_48h"]
    model = DeteriorationScore(config).fit(table, labels)
    scored = model.score(table)
    assert scored["ews_score"].between(0, 100).all()
    assert set(scored["band"]).issubset({"low", "medium", "high"})
    assert model.evaluate(table, labels).auroc > 0.7


def test_readmission_allocation_respects_capacity(cohort) -> None:
    encounters, _ = cohort
    discharges = encounters[encounters["admitted"] == 1]
    model = ReadmissionRiskModel().fit(discharges, discharges["readmitted_30d"])
    allocation = model.allocate_follow_up(discharges)
    counts = allocation["follow_up"].value_counts()
    assert counts.get("community_visit", 0) == round(len(discharges) * 0.10)
    assert allocation["readmission_risk"].is_monotonic_decreasing


def test_note_classifier_codes_common_presentations(cohort) -> None:
    encounters, _ = cohort
    model = TriageNoteClassifier().fit(encounters["triage_note"], encounters["symptom_code"])
    coded = model.code(["crushing chest pain at rest", "short of breath with wheeze"])
    assert list(coded["symptom_code"]) == ["CHEST_PAIN", "DYSPNOEA"]
    assert normalise_note("Pt c/o SOB") == "patient complains of short of breath"


def test_length_of_stay_forecast_shape(cohort) -> None:
    encounters, _ = cohort
    admissions = encounters[encounters["admitted"] == 1]
    model = LengthOfStayModel().fit(admissions, admissions["length_of_stay_days"])
    assert (model.predict(admissions) >= 0).all()
    start = pd.to_datetime(admissions["arrival_time"]).max()
    forecast = model.forecast_bed_demand(admissions, start, days=5)
    assert len(forecast) == 5
    assert forecast["expected_occupied_beds"].is_monotonic_decreasing
