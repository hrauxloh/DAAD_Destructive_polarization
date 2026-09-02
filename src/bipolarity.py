"""Three simple, automated, CPU-only proxies for how much a news article
frames an issue as strictly two opposing sides ("bipolarity"). None of
these use an LLM -- they are lexicon/statistics-based, meant as quick,
transparent complements to (not replacements for) the LLM-based
black_and_white technique detection in src/prompting.py.

1. dichotomy_marker_density  -- keyword/pattern density of EXPLICIT binary
   framing language ("either...or", "vs", "two sides", ...). Cheapest,
   most transparent; only catches explicit phrasing, misses implicit
   two-sided framing (e.g. quoting two opposed officials with no "either").

2. entity_sentiment_gap -- sentiment-analysis proxy for IMPLICIT bipolar
   framing: finds the two most-mentioned named entities/actors, scores the
   sentiment of the sentences mentioning each (VADER, lexicon-based, no
   training needed), and reports the gap between them. A large gap (one
   entity framed positively, the other negatively) suggests the article is
   structuring the issue as two opposed sides even without explicit
   binary language.

3. antonym_cooccurrence_density -- density of WordNet antonym pairs that
   both appear in the article (e.g. "safe"/"dangerous"), per 1000 words.
   High density suggests the text repeatedly sets up oppositional word
   pairs.

None of these are validated against human-labeled bipolarity judgments in
this project yet -- they're transparent, cheap proxies for comparison, not
a claimed ground truth.
"""

import re
from collections import Counter

from nltk.corpus import wordnet as wn
from nltk.sentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# 1. Dichotomy marker density
# ---------------------------------------------------------------------------

_DICHOTOMY_PHRASES = [
    "on one side", "on the other side", "on the other hand",
    "one side", "other side", "for or against",
    "us vs", "us versus", "them vs", "them versus",
    "two camps", "two sides", "one or the other",
    "black and white", "no middle ground", "with us or against us",
]
_VS_PATTERN = re.compile(r"\bvs\.?\b|\bversus\b", re.IGNORECASE)
_EITHER_OR_PATTERN = re.compile(r"\beither\b.{0,80}?\bor\b", re.IGNORECASE | re.DOTALL)


def dichotomy_marker_density(text: str) -> dict:
    words = re.findall(r"\w+", text)
    n_words = len(words)
    if n_words == 0:
        return {"dichotomy_marker_count": 0, "dichotomy_marker_density": None}

    lower = text.lower()
    count = sum(lower.count(phrase) for phrase in _DICHOTOMY_PHRASES)
    count += len(_VS_PATTERN.findall(text))
    count += len(_EITHER_OR_PATTERN.findall(text))

    return {
        "dichotomy_marker_count": count,
        "dichotomy_marker_density": 1000 * count / n_words,
    }


# ---------------------------------------------------------------------------
# 2. Entity sentiment polarity gap
# ---------------------------------------------------------------------------

_sia = None


def _get_sia():
    global _sia
    if _sia is None:
        _sia = SentimentIntensityAnalyzer()
    return _sia


def entity_sentiment_gap(doc) -> dict:
    """`doc` is a spaCy Doc (caller passes one in so the pipeline can reuse
    a single nlp() call per article across all three metrics)."""
    sia = _get_sia()
    sentences = list(doc.sents)

    entity_counts = Counter(
        ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE", "NORP")
    )
    top_entities = [e for e, _ in entity_counts.most_common(2)]

    if len(top_entities) < 2:
        return {
            "entity_1": top_entities[0] if top_entities else None,
            "entity_2": None,
            "entity_1_sentiment": None,
            "entity_2_sentiment": None,
            "entity_sentiment_gap": None,
        }

    def mean_sentiment_for(entity):
        scores = [
            sia.polarity_scores(sent.text)["compound"]
            for sent in sentences
            if entity in sent.text
        ]
        return sum(scores) / len(scores) if scores else None

    s1 = mean_sentiment_for(top_entities[0])
    s2 = mean_sentiment_for(top_entities[1])
    gap = abs(s1 - s2) if s1 is not None and s2 is not None else None

    return {
        "entity_1": top_entities[0],
        "entity_2": top_entities[1],
        "entity_1_sentiment": s1,
        "entity_2_sentiment": s2,
        "entity_sentiment_gap": gap,
    }


# ---------------------------------------------------------------------------
# 3. WordNet antonym co-occurrence density
# ---------------------------------------------------------------------------


def antonym_cooccurrence_density(doc) -> dict:
    words = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
    n_words = len(words)
    if n_words == 0:
        return {"antonym_pair_count": 0, "antonym_cooccurrence_density": None, "antonym_pairs_sample": []}

    word_set = set(words)
    seen_pairs = set()
    for w in word_set:
        for syn in wn.synsets(w):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    ant_word = ant.name().lower()
                    if ant_word in word_set and ant_word != w:
                        seen_pairs.add(tuple(sorted((w, ant_word))))

    count = len(seen_pairs)
    return {
        "antonym_pair_count": count,
        "antonym_cooccurrence_density": 1000 * count / n_words,
        "antonym_pairs_sample": sorted(seen_pairs)[:10],
    }


# ---------------------------------------------------------------------------
# Combined driver
# ---------------------------------------------------------------------------


def compute_bipolarity_row(document_id: str, title: str, text: str, nlp) -> dict:
    doc = nlp(text)
    row = {"document_id": document_id, "title": title}
    row.update(dichotomy_marker_density(text))
    row.update(entity_sentiment_gap(doc))
    row.update(antonym_cooccurrence_density(doc))
    return row


def compute_bipolarity_table(articles: list[dict], nlp=None) -> list[dict]:
    """`articles` is a list of dicts each with document_id, title, full_text
    keys. Loads spaCy once (unless `nlp` is passed in) and reuses it."""
    if nlp is None:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    return [
        compute_bipolarity_row(a["document_id"], a["title"], a["full_text"], nlp)
        for a in articles
    ]
