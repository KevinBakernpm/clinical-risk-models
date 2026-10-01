# clinical-risk-models

Reference implementation of five machine learning models used for clinical
decision support across the emergency department (ED) and acute inpatient
wards: triage acuity, early warning of deterioration, 30-day readmission
risk, symptom coding of triage notes, and length-of-stay / bed demand
forecasting.

The code is written for clarity and reproducibility with plain
scikit-learn pipelines, so each model can be read, retrained and audited in
isolation. It ships with a deterministic synthetic data generator and does
not need access to any patient records.

> **Not for clinical use without local validation.** These models are
> trained on synthetic data. Any deployment requires retraining on local
> data, prospective validation, clinical safety sign-off and ongoing
> performance monitoring.

## Models and where they sit in the clinical workflow

| Module | Algorithm | Decision supported | Used by |
| --- | --- | --- | --- |
| `triage_acuity.py` | Gradient boosting classifier | Five-level triage acuity for every adult ED arrival; sets queue position and flags disagreements with nurse triage for senior review | ED triage nurses, ED coordinator |
| `deterioration_score.py` | Logistic regression over windowed ward observations | 0-100 early warning score for sepsis / deterioration within 48 h; high band pages the critical care outreach team | Ward nurses, outreach team |
| `readmission_risk.py` | Random forest classifier | 30-day emergency readmission risk at discharge; allocates community visits and follow-up calls within team capacity | Discharge coordinators, transitional care team |
| `note_classifier.py` | TF-IDF + linear SVM | Codes the free-text triage note into a presenting-symptom category and selects the clinical pathway; low-margin notes go to manual coding | ED record, clinical coders, liaison psychiatry |
| `length_of_stay.py` | Histogram gradient boosting regressor | Predicted length of stay and expected discharge date per admission, rolled up into a 7-day bed occupancy forecast | Site operations / bed management |

### Inputs

All models draw on a common encounter record:

* **Demographics** - age, sex, ethnicity, area deprivation decile
* **Arrival** - arrival mode, presenting complaint, free-text triage note
* **Vital signs** - heart rate, respiratory rate, blood pressure,
  temperature, SpO2, GCS, pain score
* **Labs** - lactate, white cell count, creatinine, CRP
* **History** - comorbidity count, admissions in the previous 12 months
* **Ward observations** - four-hourly vitals and lactate during admission
  (deterioration score only)

## Architecture

```
            encounters + observations (synthetic or local extract)
                               |
                       features.py (impute, encode, windowing)
       +-----------+-----------+-----------+-----------+
       |           |           |           |           |
   triage      deterioration readmission  note       length of
   acuity      score         risk         classifier stay
       |           |           |           |           |
   ED queue   escalation    follow-up    pathway     bed demand
              band          tier         + coding    forecast
```

Each model is a self-contained class wrapping a scikit-learn `Pipeline`,
with `fit`, a prediction method, an operational method that turns
predictions into a workflow decision (`triage_queue`, `score`,
`allocate_follow_up`, `code`, `forecast_bed_demand`) and an `evaluate`
method returning the metrics the clinical team monitors. Hyperparameters
and operating thresholds live in `config.py`.

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Generate a synthetic cohort, train all five models, print metrics and examples
python run_pipeline.py

# Optional: write the synthetic data to CSV
python data/generate_synthetic.py --n-encounters 4000 --out data/raw

# Tests
pip install pytest
python -m pytest -q
```

A full run takes around a minute on a laptop.

## Data

All data is **synthetic**. `data/generate_synthetic.py` simulates ED
attendances and ward observation charts from a fixed seed, with outcomes
derived from plausible clinical relationships (for example, rising lactate
and respiratory rate preceding deterioration). The same seed always
produces the same cohort, so metrics are reproducible.

## Project structure

```
clinical-risk-models/
├── data/
│   └── generate_synthetic.py      # deterministic synthetic cohort
├── src/clinical_risk/
│   ├── config.py                  # feature sets, hyperparameters, thresholds
│   ├── features.py                # preprocessing and observation windowing
│   ├── triage_acuity.py           # ED triage acuity (gradient boosting)
│   ├── deterioration_score.py     # early warning score (logistic regression)
│   ├── readmission_risk.py        # 30-day readmission (random forest)
│   ├── note_classifier.py         # triage note coding (TF-IDF + linear SVM)
│   ├── length_of_stay.py          # LOS and bed demand (gradient boosting)
│   └── pipeline.py                # train / evaluate / score orchestration
├── tests/
├── run_pipeline.py                # entry point
└── requirements.txt
```

## License

MIT - see [LICENSE](LICENSE).
