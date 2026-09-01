"""Combines src/readability.py (Flesch-Kincaid and related indices) and
src/cohmetrix_lite.py (open Coh-Metrix-inspired approximation) into one
per-article results table. Pure text-statistics computation -- no LLM, no
GPU needed, so this can run quickly over the full corpus."""

import spacy

from .cohmetrix_lite import compute_cohmetrix_lite
from .readability import compute_readability


def compute_complexity_row(document_id: str, title: str, text: str, nlp=None) -> dict:
    row = {"document_id": document_id, "title": title}
    row.update(compute_readability(text))
    row.update(compute_cohmetrix_lite(text, nlp=nlp))
    return row


def compute_complexity_table(articles: list[dict]) -> list[dict]:
    """`articles` is a list of dicts each with at least document_id, title,
    full_text keys (matching australia_498sample_climatechange.csv). Loads
    the spaCy model once and reuses it across all articles."""
    nlp = spacy.load("en_core_web_sm")
    rows = []
    for article in articles:
        rows.append(
            compute_complexity_row(
                article["document_id"], article["title"], article["full_text"], nlp=nlp
            )
        )
    return rows
