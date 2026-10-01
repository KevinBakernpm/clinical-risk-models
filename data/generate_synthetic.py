"""Deterministic synthetic patient data for the clinical-risk-models pipeline.

Generates two tables that mirror the shape of a typical acute-care data extract:

* ``encounters`` - one row per emergency department attendance with
  demographics, triage vital signs, first-draw labs, the free-text triage
  note, and downstream outcomes (acuity, admission, length of stay,
  48-hour deterioration and 30-day readmission).
* ``observations`` - repeated ward observations (vital signs and lactate)
  recorded every four hours during the first 24 hours of each admission.

All values are simulated from a fixed random seed. No real patient data is
used or required. Outcomes are generated from plausible clinical
relationships so that the downstream models have signal to learn.

Usage::

    python data/generate_synthetic.py --n-encounters 4000 --out data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ETHNICITY_GROUPS = ["white", "asian", "black", "mixed", "other"]
ETHNICITY_WEIGHTS = [0.72, 0.12, 0.07, 0.04, 0.05]
ARRIVAL_MODES = ["walk_in", "ambulance", "gp_referral"]

# Presenting complaint -> (symptom code, baseline severity, note phrases)
COMPLAINTS: dict[str, tuple[str, float, list[str]]] = {
    "chest_pain": ("CHEST_PAIN", 1.2, [
        "central chest pain radiating to left arm",
        "tight chest since this morning with sweating",
        "crushing retrosternal pain at rest",
        "pleuritic chest pain worse on deep breath",
        "chest heaviness on exertion relieved by rest",
    ]),
    "shortness_of_breath": ("DYSPNOEA", 1.1, [
        "short of breath on minimal exertion",
        "increasing breathlessness over three days with wheeze",
        "unable to complete sentences, using accessory muscles",
        "productive cough and shortness of breath",
        "known copd, worsening dyspnoea and green sputum",
    ]),
    "abdominal_pain": ("ABDO_PAIN", 0.7, [
        "right iliac fossa pain with nausea",
        "epigastric pain after eating, vomiting twice",
        "generalised abdominal pain and distension",
        "colicky loin to groin pain",
        "lower abdominal cramps with loose stools",
    ]),
    "head_injury": ("HEAD_INJURY", 0.8, [
        "fall striking head on pavement, brief loss of consciousness",
        "laceration to forehead after assault",
        "headache and vomiting following head knock",
        "hit head on cupboard door, no loc, on anticoagulant",
    ]),
    "fall": ("FALL_MSK", 0.4, [
        "mechanical fall at home, painful right hip",
        "twisted ankle playing football, unable to weight bear",
        "fall on outstretched hand, swollen wrist",
        "found on floor by carer, long lie overnight",
        "back pain after lifting, no red flags",
    ]),
    "fever": ("FEVER_INFECTION", 0.9, [
        "fever and rigors for two days",
        "temperature at home, feels hot and shivery, dysuria",
        "cellulitis spreading up lower leg",
        "cough fever and feeling generally unwell",
        "confused and febrile per nursing home staff",
    ]),
    "collapse": ("SYNCOPE", 1.0, [
        "collapsed in supermarket, witnessed brief loc",
        "dizzy then blacked out on standing",
        "syncopal episode with palpitations",
        "near faint while queuing, now feels well",
    ]),
    "mental_health": ("MENTAL_HEALTH", 0.5, [
        "low mood and thoughts of self harm",
        "anxious, chest tight, panic attack history",
        "brought in by police under section, agitated",
        "intentional overdose of paracetamol tablets",
        "hearing voices, not sleeping, family concerned",
    ]),
}
COMPLAINT_WEIGHTS = [0.14, 0.12, 0.15, 0.08, 0.17, 0.13, 0.09, 0.12]
NOTE_FILLERS = [
    "pt reports", "per paramedics", "states", "c/o", "history of",
    "seen by triage nurse", "obs taken", "no allergies known", "lives alone",
    "pmh htn", "pmh t2dm", "ambulance crew handover", "",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _triage_note(rng: np.random.Generator, complaint: str) -> str:
    phrases = COMPLAINTS[complaint][2]
    parts = [rng.choice(NOTE_FILLERS), rng.choice(phrases)]
    if rng.random() < 0.35:
        parts.append(rng.choice(NOTE_FILLERS))
    # Occasional cross-talk so notes are not perfectly separable.
    if rng.random() < 0.08:
        other = rng.choice(list(COMPLAINTS))
        parts.append(rng.choice(COMPLAINTS[other][2]).split(",")[0])
    return " ".join(p for p in parts if p).strip()


def generate_encounters(n_encounters: int = 4000, seed: int = 20260) -> pd.DataFrame:
    """Simulate ``n_encounters`` emergency department attendances."""
    rng = np.random.default_rng(seed)
    n = n_encounters

    age = np.clip(rng.gamma(shape=4.0, scale=13.0, size=n), 16, 101).round().astype(int)
    sex = rng.choice(["female", "male"], size=n, p=[0.52, 0.48])
    ethnicity = rng.choice(ETHNICITY_GROUPS, size=n, p=ETHNICITY_WEIGHTS)
    imd_decile = rng.integers(1, 11, size=n)  # 1 = most deprived
    complaint = rng.choice(list(COMPLAINTS), size=n, p=COMPLAINT_WEIGHTS)
    base_sev = np.array([COMPLAINTS[c][1] for c in complaint])

    comorbidity_count = rng.poisson(lam=0.4 + age / 35.0)
    prior_admissions_12m = rng.poisson(lam=0.15 + comorbidity_count * 0.25 + (11 - imd_decile) * 0.03)

    # Latent illness severity drives physiology and outcomes.
    severity = (
        base_sev
        + 0.018 * (age - 50)
        + 0.15 * comorbidity_count
        + rng.normal(0, 0.8, size=n)
    )
    arrival_logit = -0.8 + 0.9 * severity + 0.02 * (age - 50)
    ambulance = rng.random(n) < _sigmoid(arrival_logit)
    arrival_mode = np.where(ambulance, "ambulance",
                            rng.choice(["walk_in", "gp_referral"], size=n, p=[0.75, 0.25]))

    heart_rate = rng.normal(82 + 9 * severity, 10, n).clip(35, 190).round()
    resp_rate = rng.normal(16 + 2.2 * severity, 2.5, n).clip(8, 45).round()
    sbp = rng.normal(132 - 9 * np.maximum(severity - 1, 0), 18, n).clip(60, 230).round()
    dbp = (sbp * rng.normal(0.6, 0.05, n)).clip(30, 140).round()
    temperature = rng.normal(36.9 + 0.35 * severity * (complaint == "fever") + 0.1 * severity, 0.5, n)
    temperature = temperature.clip(34.0, 41.5).round(1)
    spo2 = (98.5 - 1.6 * np.maximum(severity, 0) * np.isin(complaint, ["shortness_of_breath", "fever"])
            - 0.6 * np.maximum(severity, 0) + rng.normal(0, 1.2, n)).clip(70, 100).round()
    gcs = np.where(rng.random(n) < _sigmoid(severity - 3.5),
                   rng.integers(8, 15, size=n), 15)
    pain_score = rng.integers(0, 11, size=n)

    lactate = rng.lognormal(np.log(1.1) + 0.22 * np.maximum(severity, 0), 0.35, n).round(1)
    wbc = rng.normal(8.0 + 2.5 * severity * (complaint == "fever") + 0.8 * severity, 2.2, n)
    wbc = wbc.clip(1.0, 40.0).round(1)
    creatinine = rng.normal(78 + 0.6 * (age - 50) + 9 * severity, 18, n).clip(30, 600).round()
    crp = rng.lognormal(np.log(8) + 0.6 * np.maximum(severity, 0) + 0.8 * (complaint == "fever"), 0.8, n)
    crp = crp.clip(0.5, 450).round()

    # Acuity 1 (resuscitation) .. 5 (non-urgent), assigned by experienced triage.
    acuity_signal = severity + 0.3 * (gcs < 15) + 0.4 * (spo2 < 92) + rng.normal(0, 0.45, n)
    acuity = 5 - np.digitize(acuity_signal, bins=[0.3, 1.2, 2.2, 3.2])
    acuity = acuity.astype(int)

    admitted = rng.random(n) < _sigmoid(-1.9 + 1.25 * severity + 0.15 * comorbidity_count)
    los_mean = np.exp(0.6 + 0.35 * np.maximum(severity, 0) + 0.012 * (age - 50)
                      + 0.08 * comorbidity_count)
    length_of_stay_days = np.where(
        admitted, rng.gamma(shape=3.0, scale=los_mean / 3.0).clip(0.3, 60).round(1), 0.0
    )

    deteriorated_48h = admitted & (
        rng.random(n) < _sigmoid(-4.2 + 0.9 * severity + 0.55 * (lactate - 1.5) + 0.01 * (age - 50))
    )
    readmit_logit = (
        -3.1 + 0.018 * (age - 50) + 0.28 * comorbidity_count + 0.45 * prior_admissions_12m
        + 0.06 * (10 - imd_decile) + 0.04 * length_of_stay_days
    )
    readmitted_30d = admitted & (rng.random(n) < _sigmoid(readmit_logit))

    notes = [_triage_note(rng, c) for c in complaint]
    symptom_code = [COMPLAINTS[c][0] for c in complaint]

    arrival_time = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        np.sort(rng.uniform(0, 365 * 24, n)), unit="h"
    )

    return pd.DataFrame({
        "encounter_id": [f"E{100000 + i}" for i in range(n)],
        "arrival_time": arrival_time.round("min"),
        "age": age,
        "sex": sex,
        "ethnicity": ethnicity,
        "imd_decile": imd_decile,
        "arrival_mode": arrival_mode,
        "presenting_complaint": complaint,
        "heart_rate": heart_rate,
        "resp_rate": resp_rate,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "temperature": temperature,
        "spo2": spo2,
        "gcs": gcs,
        "pain_score": pain_score,
        "comorbidity_count": comorbidity_count,
        "prior_admissions_12m": prior_admissions_12m,
        "lactate": lactate,
        "wbc": wbc,
        "creatinine": creatinine,
        "crp": crp,
        "triage_note": notes,
        "symptom_code": symptom_code,
        "acuity": acuity,
        "admitted": admitted.astype(int),
        "length_of_stay_days": length_of_stay_days,
        "deteriorated_48h": deteriorated_48h.astype(int),
        "readmitted_30d": readmitted_30d.astype(int),
    })


def generate_observations(encounters: pd.DataFrame, seed: int = 20260,
                          interval_hours: int = 4, horizon_hours: int = 24) -> pd.DataFrame:
    """Simulate ward observation charts for admitted encounters.

    Patients who go on to deteriorate show a gradual drift in heart rate,
    respiratory rate, blood pressure, oxygen saturation and lactate.
    """
    rng = np.random.default_rng(seed + 1)
    admitted = encounters[encounters["admitted"] == 1]
    hours = np.arange(0, horizon_hours, interval_hours)
    rows: list[dict[str, object]] = []
    for enc in admitted.itertuples(index=False):
        drift = 1.0 if enc.deteriorated_48h else 0.0
        for step, h in enumerate(hours):
            frac = step / max(len(hours) - 1, 1)
            rows.append({
                "encounter_id": enc.encounter_id,
                "hours_since_admission": int(h),
                "heart_rate": round(enc.heart_rate + drift * 14 * frac + rng.normal(0, 6)),
                "resp_rate": round(enc.resp_rate + drift * 5 * frac + rng.normal(0, 1.5)),
                "systolic_bp": round(enc.systolic_bp - drift * 18 * frac + rng.normal(0, 8)),
                "temperature": round(enc.temperature + drift * 0.6 * frac + rng.normal(0, 0.3), 1),
                "spo2": round(min(100.0, enc.spo2 - drift * 3 * frac + rng.normal(0, 1.0))),
                "lactate": round(max(0.3, enc.lactate + drift * 1.2 * frac + rng.normal(0, 0.2)), 1),
            })
    return pd.DataFrame(rows)


def generate_cohort(n_encounters: int = 4000, seed: int = 20260) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(encounters, observations)`` for a reproducible synthetic cohort."""
    encounters = generate_encounters(n_encounters=n_encounters, seed=seed)
    observations = generate_observations(encounters, seed=seed)
    return encounters, observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-encounters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "raw")
    args = parser.parse_args()

    encounters, observations = generate_cohort(args.n_encounters, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    encounters.to_csv(args.out / "encounters.csv", index=False)
    observations.to_csv(args.out / "observations.csv", index=False)
    print(f"Wrote {len(encounters)} encounters and {len(observations)} observations to {args.out}")


if __name__ == "__main__":
    main()
