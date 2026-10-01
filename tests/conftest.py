from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_synthetic import generate_cohort  # noqa: E402


@pytest.fixture(scope="session")
def cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_cohort(n_encounters=1500, seed=7)
