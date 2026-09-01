"""An OPEN-SOURCE APPROXIMATION of a subset of Coh-Metrix indices.

Coh-Metrix itself (Graesser, McNamara et al.) is a closed tool: there is no
public API or downloadable package, only a web form / Windows desktop app
that requires manually submitting one text at a time. It cannot be wired
into an automated batch pipeline. What follows instead is a set of
spaCy-based metrics INSPIRED BY Coh-Metrix's cohesion/complexity
categories, computed with simple, transparent heuristics:

  - referential_cohesion: lexical (lemma) overlap between adjacent
    sentences, as a proxy for Coh-Metrix's "argument overlap" /
    co-reference indices. Coh-Metrix itself additionally uses LSA
    (latent semantic analysis) for a deeper semantic-similarity version
    of this; this is the surface-lexical version only.
  - causal / logical / temporal / additive connective density: counts of
    connective words per 1000 tokens, matching Coh-Metrix's connective
    categories, using a hand-built word list (not Coh-Metrix's exact list,
    which is not public).
  - mean_dependency_depth: average maximum dependency-tree depth per
    sentence, as a general syntactic-complexity proxy (Coh-Metrix instead
    reports things like "words before the main verb" and specific
    syntactic similarity indices).

Treat these as a reasonable, cite-able approximation for research use, NOT
as equivalent to (or validated against) the official Coh-Metrix tool. If
exact Coh-Metrix scores are required, texts must be run through the actual
tool (http://tool.cohmetrix.com or the desktop app) by hand.
"""

import spacy

_CONNECTIVES = {
    "causal": {
        "because", "since", "therefore", "thus", "so", "consequently",
        "hence", "accordingly",
    },
    "logical": {
        "however", "although", "though", "but", "nevertheless", "whereas",
        "yet", "nonetheless", "despite",
    },
    "temporal": {
        "before", "after", "when", "while", "until", "then", "meanwhile",
        "since", "subsequently",
    },
    "additive": {
        "and", "also", "moreover", "furthermore", "besides", "additionally",
    },
}
# Multi-word connectives checked separately against the raw lowercased text.
_MULTIWORD_CONNECTIVES = {
    "causal": ["as a result", "due to", "because of"],
    "logical": ["on the other hand", "in contrast", "even though"],
    "temporal": ["as soon as", "at the same time"],
    "additive": ["in addition", "on top of that"],
}

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _dependency_depth(token) -> int:
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(_dependency_depth(c) for c in children)


def compute_cohmetrix_lite(text: str, nlp=None) -> dict:
    """Returns a dict of spaCy-based Coh-Metrix-inspired indices for one
    article-length text."""
    nlp = nlp or _get_nlp()
    doc = nlp(text)
    sentences = list(doc.sents)
    n_sentences = len(sentences)
    tokens = [t for t in doc if not t.is_space]
    n_tokens = len(tokens)

    if n_sentences == 0 or n_tokens == 0:
        return {
            "referential_cohesion": None,
            "causal_connective_density": None,
            "logical_connective_density": None,
            "temporal_connective_density": None,
            "additive_connective_density": None,
            "mean_dependency_depth": None,
        }

    # Referential cohesion: content-lemma overlap between adjacent sentences.
    content_lemma_sets = [
        {t.lemma_.lower() for t in sent if t.pos_ in ("NOUN", "PROPN", "VERB", "ADJ")}
        for sent in sentences
    ]
    overlaps = []
    for a, b in zip(content_lemma_sets, content_lemma_sets[1:]):
        if not a or not b:
            continue
        shared = len(a & b)
        denom = (len(a) + len(b)) / 2
        overlaps.append(shared / denom if denom else 0.0)
    referential_cohesion = sum(overlaps) / len(overlaps) if overlaps else 0.0

    # Connective density per 1000 tokens, by category.
    lower_tokens = [t.text.lower() for t in tokens]
    lower_text = text.lower()
    densities = {}
    for category, word_set in _CONNECTIVES.items():
        count = sum(1 for w in lower_tokens if w in word_set)
        for phrase in _MULTIWORD_CONNECTIVES[category]:
            count += lower_text.count(phrase)
        densities[f"{category}_connective_density"] = 1000 * count / n_tokens

    # Mean dependency-tree depth per sentence.
    depths = [_dependency_depth(sent.root) for sent in sentences]
    mean_dependency_depth = sum(depths) / len(depths)

    return {
        "referential_cohesion": referential_cohesion,
        **densities,
        "mean_dependency_depth": mean_dependency_depth,
    }
