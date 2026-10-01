"""Generate a synthetic cohort, train every model and print held-out metrics.

Usage::

    python run_pipeline.py --n-encounters 4000 --seed 20260
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from clinical_risk.pipeline import run_all  # noqa: E402
from generate_synthetic import generate_cohort  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the clinical risk models.")
    parser.add_argument("--n-encounters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260)
    args = parser.parse_args()

    encounters, observations = generate_cohort(args.n_encounters, args.seed)
    run_all(encounters, observations)


if __name__ == "__main__":
    main()
