# DAAD_Destructive_polarization

## Complexity-erasing language detection proof of concept

Detects 2 techniques that erase complexity in public debate (a subset of the
SemEval / Da San Martino et al. propaganda taxonomy): `black_and_white`
(dichotomous reasoning) and `thought_terminating_cliche`. Uses a **two-step,
article-level** pipeline with Llama 3.1 8B Instruct run locally (no paid API)
via `llama-cpp-python`, sized to run on a free-tier Google Colab T4 GPU.

For each article, two separate model calls:
1. **Claim extraction (Step 1)** — read the whole article and list the
   discrete claims/assertions it makes
2. **Instance extraction (Step 2)** — given the article and the Step 1
   claims, pull out every concrete instance of the 2 techniques anywhere in
   the article: an exact quote, the claim it operates on, and a rationale,
   guided by explicit "distinction from X" rules in the codebook (e.g. a
   statement that rejects binary blame is NOT the black-and-white fallacy; a
   short but substantive sentence is NOT a thought-terminating cliché). The
   same technique can be reported more than once per article.

This design evolved from an earlier flat free-text extraction prompt (which
under-flagged isolated span fragments and over-flagged ordinary prose) to a
per-paragraph two-step YES/NO version, and now to this article-level,
instance-extracting version so Step 1 sees the whole article's argument
rather than one paragraph at a time, and Step 2 reports every match rather
than a single yes/no per chunk.

### Data
- `oversimplification_data.csv` — ~190 SemEval-derived spans labeled
  `Black-and-White_Fallacy` or `Thought-terminating_Cliches` (rows with other
  labels are skipped). No article full-text is included, so `start`/`end`
  offsets aren't resolvable here; eval runs each `span_text` through the
  pipeline as its own standalone "article" — deprioritized for now, see
  `src/eval.py`.
- `australia_498sample_climatechange.csv` — ~480 full news articles (not
  labeled for propaganda), the current priority target for running the
  extraction pipeline.
- `aus_sample_preprocessing.R` — an R script that splits the climate-change
  corpus into per-sentence rows (a separate, sentence-level preprocessing
  path not used by the current Python pipeline, which operates on whole
  articles).

### Layout
- `src/codebook.py` — technique definitions, "distinction from X" guardrails,
  few-shot examples, and the mapping from `oversimplification_data.csv`
  technique labels to internal keys
- `src/prompting.py` — the two-step (claim extraction, then instance
  extraction) prompt construction, JSON-output parsing/validation, and the
  `analyze_article()` driver that runs both calls
- `src/paragraphs.py` — splits article full-text into paragraph-sized chunks
  (used by an earlier per-paragraph version of the pipeline; not used by the
  current article-level notebook flow)
- `src/eval.py` — loads `oversimplification_data.csv` and evaluates the
  pipeline against it
- `notebooks/colab_propaganda_poc.ipynb` — the runnable Colab notebook (model
  download, two-step pipeline, evaluation, and an article-level run over the
  climate-change corpus, saving results to CSV)
- `src/readability.py` — standard readability metrics (Flesch-Kincaid Grade
  Level, Flesch Reading Ease, Gunning Fog, Coleman-Liau, ARI) via the
  well-established `textstat` package
- `src/cohmetrix_lite.py` — an **open-source approximation** of a subset of
  Coh-Metrix's cohesion/complexity indices (referential cohesion, connective
  density by category, syntactic depth), computed with spaCy. **Not the
  official Coh-Metrix tool** — see the module docstring for why (Coh-Metrix
  has no public API/package, only a manual web form / desktop app) and
  exactly what is/isn't approximated
- `src/complexity_pipeline.py` — combines the two into one per-article
  results table; no LLM or GPU needed
- `notebooks/colab_complexity_pipeline.ipynb` — CPU-only Colab notebook that
  runs the complexity pipeline over the full climate-change corpus and saves
  `aus_complexity_scores.csv`

### Running the PoC
Open `notebooks/colab_propaganda_poc.ipynb` in Google Colab (`Runtime > Change
runtime type > T4 GPU`) and run the cells top to bottom. It clones this repo
(must be public, or you handle auth yourself), downloads a public GGUF
conversion of Llama-3.1-8B-Instruct (no Hugging Face token required), and runs
the two-step pipeline over each article. Section 8 saves `aus_claims.csv`
(one row per article) and `aus_instances.csv` (one row per detected
technique instance) and offers them for download.

### Known limitations
- Currently scoped to 2 techniques by request (`black_and_white`,
  `thought_terminating_cliche`); `causal_oversimplification` and
  `reductio_ad_hitlerum` were previously covered and can be re-added to
  `src/codebook.py` if needed later.
- Article text is truncated (`MAX_ARTICLE_CHARS` in the notebook) to fit the
  model's context budget on a free T4, so instances in a truncated tail
  won't be found.
- Step 2 now has to scan a whole article and report every instance, a harder
  task for an 8B model than a single paragraph YES/NO — check
  `aus_instances.csv` for both missed and spurious instances before trusting
  it at scale.
- `oversimplification_data.csv` has no non-propaganda ("none") examples, so
  eval there only measures recall/technique-confusion, not false-positive
  rate.
- An 8B quantized model is less reliable than a larger hosted model at strict
  JSON formatting and nuanced technique judgments.
- `cohmetrix_lite` is a stand-in, not a validated equivalent to Coh-Metrix —
  don't report its output as official Coh-Metrix scores. Flesch-Kincaid and
  related formulas measure surface reading difficulty, not argumentative
  complexity, so they answer a different question than the propaganda-
  technique coding pipeline above.

### Running the complexity pipeline
**In Colab:** open `notebooks/colab_complexity_pipeline.ipynb` (no GPU
needed — any runtime works) and run the cells top to bottom. It clones this
repo, installs `textstat` and spaCy's small English model, runs both metric
families over every article in `australia_498sample_climatechange.csv`
(takes a few minutes for the full ~480 articles), and saves/downloads
`aus_complexity_scores.csv` with one row per article.

**Locally:** `scripts/run_complexity.py` does the same thing outside Colab.
From the repo root:
```
pip install textstat spacy pandas nltk
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('cmudict')"
python scripts/run_complexity.py
```
This is CPU-only and lightweight (no LLM, no GPU) — tested locally at
~0.4 sec/article, so the full corpus finishes in a few minutes on any
machine. Output CSVs from either path (`aus_complexity_scores.csv`,
`aus_claims.csv`, `aus_instances.csv`) are gitignored as generated
artifacts, not committed to the repo.

**Plotting the results:** once `aus_complexity_scores.csv` exists (from
either path above), run:
```
pip install matplotlib
python scripts/plot_complexity.py
```
This produces `complexity_by_publisher.png` — a grid of strip plots (one
per complexity index), each article a point colored/shaped by publisher
(top 7 by article count shown individually, the rest grouped as "Other"),
with a black tick marking each publisher's median. Axes are scaled to the
1st–99th percentile per index so a rare degenerate article (e.g. a
sentence-splitting failure inflating a readability score) doesn't compress
the rest of the distribution into an unreadable band — such outliers still
plot, just clipped at the axis edge, and the figure caption reports how
many were clipped.
