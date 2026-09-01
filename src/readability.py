"""Standard readability metrics (Flesch-Kincaid and related), via the
well-established `textstat` package. These are surface-level reading-
difficulty measures (based on sentence length and syllable counts) -- they
do NOT measure argumentative or conceptual complexity. See
src/cohmetrix_lite.py for indices closer to that construct.
"""

import textstat


def compute_readability(text: str) -> dict:
    """Returns a dict of readability scores for one text (article-length
    input is fine; these formulas don't need a minimum length the way some
    other readability indices, like SMOG, do)."""
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "word_count": textstat.lexicon_count(text, removepunct=True),
        "sentence_count": textstat.sentence_count(text),
    }
