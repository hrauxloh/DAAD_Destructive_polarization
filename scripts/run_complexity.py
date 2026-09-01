"""Runs the linguistic complexity pipeline (Flesch-Kincaid + Coh-Metrix-lite,
see src/readability.py and src/cohmetrix_lite.py) over
australia_498sample_climatechange.csv and saves aus_complexity_scores.csv.

CPU-only, no LLM/GPU needed. Run from the repo root:

    pip install textstat spacy pandas nltk
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('cmudict')"
    python scripts/run_complexity.py

See notebooks/colab_complexity_pipeline.ipynb for a Colab-notebook version
of the same pipeline.
"""

import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.complexity_pipeline import compute_complexity_table  # noqa: E402

INPUT_CSV = REPO_ROOT / "australia_498sample_climatechange.csv"
OUTPUT_CSV = REPO_ROOT / "aus_complexity_scores.csv"


def main():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        articles = list(csv.DictReader(f))
    print(f"loaded {len(articles)} articles")

    rows = compute_complexity_table(articles)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"saved {len(df)} rows to {OUTPUT_CSV}")
    print(df.describe())


if __name__ == "__main__":
    main()
