"""Free-text triage note classifier for presenting-symptom coding.

Assigns a coded presenting-symptom category (for example CHEST_PAIN,
DYSPNOEA, SYNCOPE, MENTAL_HEALTH) to the free-text note written by the
triage nurse or ambulance crew. Notes are vectorised with word and bigram
TF-IDF and classified by a linear support vector machine.

The code is pre-filled in the ED record, selects the clinical pathway and
order set offered to the treating clinician, and routes mental-health
presentations to the liaison psychiatry referral queue. Notes where the
top two class scores are within ``review_margin`` are sent to a clinical
coder for manual review instead of being auto-coded.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE, NoteClassifierConfig

PATHWAYS = {
    "CHEST_PAIN": "acute coronary syndrome pathway",
    "DYSPNOEA": "respiratory assessment pathway",
    "ABDO_PAIN": "surgical assessment pathway",
    "HEAD_INJURY": "head injury CT decision rule",
    "FALL_MSK": "minor injuries stream",
    "FEVER_INFECTION": "sepsis screening pathway",
    "SYNCOPE": "transient loss of consciousness pathway",
    "MENTAL_HEALTH": "liaison psychiatry referral",
}

_ABBREVIATIONS = {
    "c/o": "complains of",
    "loc": "loss of consciousness",
    "sob": "short of breath",
    "pt": "patient",
    "pmh": "past medical history",
}


def normalise_note(text: str) -> str:
    """Lower-case a note and expand common triage abbreviations."""
    tokens = str(text).lower().replace(",", " ").split()
    return " ".join(_ABBREVIATIONS.get(tok, tok) for tok in tokens)


@dataclass
class NoteClassifierMetrics:
    accuracy: float
    macro_f1: float
    auto_coded_rate: float
    auto_coded_accuracy: float


class TriageNoteClassifier:
    """TF-IDF + linear SVM model mapping triage notes to symptom codes."""

    def __init__(self, config: NoteClassifierConfig | None = None) -> None:
        self.config = config or NoteClassifierConfig()
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=normalise_note,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                sublinear_tf=True,
            )),
            ("svm", LinearSVC(C=self.config.C, class_weight="balanced", random_state=RANDOM_STATE)),
        ])

    def fit(self, notes: Iterable[str], codes: Iterable[str]) -> "TriageNoteClassifier":
        self.pipeline.fit(list(notes), list(codes))
        return self

    def predict(self, notes: Iterable[str]) -> np.ndarray:
        return self.pipeline.predict(list(notes))

    def code(self, notes: Iterable[str]) -> pd.DataFrame:
        """Predict a symptom code, suggested pathway and whether manual review is needed."""
        notes = list(notes)
        scores = self.pipeline.decision_function(notes)
        classes = self.pipeline.classes_
        top2 = np.sort(scores, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        codes = classes[np.argmax(scores, axis=1)]
        return pd.DataFrame({
            "note": notes,
            "symptom_code": codes,
            "pathway": [PATHWAYS.get(c, "general assessment") for c in codes],
            "margin": margin.round(3),
            "needs_review": margin < self.config.review_margin,
        })

    def evaluate(self, notes: Iterable[str], codes: Iterable[str]) -> NoteClassifierMetrics:
        truth = np.asarray(list(codes))
        coded = self.code(notes)
        pred = coded["symptom_code"].to_numpy()
        auto = ~coded["needs_review"].to_numpy()
        return NoteClassifierMetrics(
            accuracy=float(accuracy_score(truth, pred)),
            macro_f1=float(f1_score(truth, pred, average="macro")),
            auto_coded_rate=float(auto.mean()),
            auto_coded_accuracy=float((pred[auto] == truth[auto]).mean()) if auto.any() else float("nan"),
        )
