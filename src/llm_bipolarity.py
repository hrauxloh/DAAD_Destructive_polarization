"""LLM-based detection of dichotomy / mutually exclusive positions in a
full article -- a single-call, article-level companion to the three
non-LLM proxies in src/bipolarity.py, so all four approaches can be
compared on the same articles.

Reuses the `black_and_white` technique definition and "distinction from X"
guardrails already established in src/codebook.py -- "a complex issue
presented as only two possible/mutually exclusive positions" is exactly
what that entry defines -- rather than duplicating the definition.

Unlike src/bipolarity.py, this is NOT CPU-only: it requires an LLM (this
project uses Llama 3.1 8B Instruct via llama-cpp-python -- see
notebooks/colab_propaganda_poc.ipynb for the model-loading setup).
"""

import json
import re

from .codebook import CODEBOOK

_TECHNIQUE = CODEBOOK["black_and_white"]

SYSTEM_PROMPT = """You are an assistant helping researchers analyze news \
articles for academic research on political polarization. You are precise \
and conservative: only report an instance when the text clearly matches \
the definition below AND none of its "distinction from X" guardrails \
apply. When in doubt, don't report it.

CONCEPT: {name}
Definition: {definition}
{distinctions}

You will be given a full news article. Find every instance anywhere in \
the article where it presents an issue as a dichotomy -- only two \
possible or mutually exclusive positions/outcomes -- matching the \
definition above. For each instance, quote the exact substring from the \
article that triggered it and give a one-sentence rationale explaining \
why the guardrails don't rule it out. The same pattern may appear more \
than once; report every distinct instance.

OUTPUT FORMAT:
Return ONLY a single JSON object (no prose, no markdown fences) shaped \
exactly like this:
{{
  "instances": [
    {{"quote": "<exact substring from the article>", "rationale": "<one sentence>"}}
  ]
}}
If no instances are found, return {{"instances": []}}.
"""

USER_TEMPLATE = """ARTICLE:
\"\"\"
{text}
\"\"\"

JSON object:"""


def _build_system_prompt() -> str:
    distinctions = "\n".join(f"- {d}" for d in _TECHNIQUE.distinctions)
    return SYSTEM_PROMPT.format(
        name=_TECHNIQUE.name,
        definition=_TECHNIQUE.definition,
        distinctions=distinctions,
    )


def build_dichotomy_messages(article_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": USER_TEMPLATE.format(text=article_text.strip())},
    ]


class ParseError(ValueError):
    pass


def _extract_json_object(raw: str) -> str:
    raw = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ParseError(f"No JSON object found in model output: {raw[:200]!r}")
    return raw[start : end + 1]


def parse_dichotomy_result(raw: str) -> list[dict]:
    candidate = _extract_json_object(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ParseError("Top-level JSON is not an object")
    instances_raw = data.get("instances")
    if not isinstance(instances_raw, list):
        raise ParseError(f"Missing/invalid 'instances' list: {data!r}")

    instances = []
    for entry in instances_raw:
        if not isinstance(entry, dict):
            raise ParseError(f"Instance is not an object: {entry!r}")
        quote = entry.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            raise ParseError(f"Missing/empty 'quote': {entry!r}")
        instances.append({"quote": quote.strip(), "rationale": entry.get("rationale")})
    return instances


def analyze_dichotomy(generate_fn, article_text: str, max_retries: int = 2) -> list[dict]:
    last_err = None
    for _ in range(max_retries):
        raw = generate_fn(build_dichotomy_messages(article_text))
        try:
            return parse_dichotomy_result(raw)
        except ParseError as e:
            last_err = e
    raise last_err


def compute_llm_dichotomy_row(
    document_id: str, title: str, text: str, generate_fn, max_retries: int = 2
) -> dict:
    word_count = len(text.split())
    try:
        instances = analyze_dichotomy(generate_fn, text, max_retries=max_retries)
        parse_error = None
    except ParseError as e:
        instances = []
        parse_error = str(e)

    density = 1000 * len(instances) / word_count if word_count else None
    return {
        "document_id": document_id,
        "title": title,
        "llm_dichotomy_instance_count": len(instances),
        "llm_dichotomy_density": density,
        "llm_dichotomy_instances": instances,
        "llm_dichotomy_parse_error": parse_error,
    }


def compute_llm_dichotomy_table(articles: list[dict], generate_fn, max_retries: int = 2) -> list[dict]:
    return [
        compute_llm_dichotomy_row(
            a["document_id"], a["title"], a["full_text"], generate_fn, max_retries=max_retries
        )
        for a in articles
    ]
