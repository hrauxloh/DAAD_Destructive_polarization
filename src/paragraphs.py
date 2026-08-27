"""Splits article full-text into paragraph-sized chunks for the per-paragraph
two-step extraction pipeline."""

import re


def split_into_paragraphs(text: str, min_chars: int = 40) -> list[str]:
    """Split on blank lines first; if that yields only one chunk (common
    once newlines have been collapsed by upstream preprocessing), fall back
    to splitting on single newlines. Drops fragments shorter than
    `min_chars` (headers, bylines, stray whitespace)."""
    text = text.strip()
    if not text:
        return []

    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    if len(chunks) <= 1:
        chunks = [c.strip() for c in text.split("\n") if c.strip()]

    return [c for c in chunks if len(c) >= min_chars]
