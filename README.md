# DAAD_Destructive_polarization

## Complexity-erasing language detection proof of concept

Detects 2 techniques that erase complexity in public debate (a subset of the
SemEval / Da San Martino et al. propaganda taxonomy): `black_and_white`
(dichotomous reasoning) and `thought_terminating_cliche`. Uses a **two-step,
per-paragraph** prompt with Llama 3.1 8B Instruct run locally (no paid API)
via `llama-cpp-python`, sized to run on a free-tier Google Colab T4 GPU.

For each paragraph:
1. **Claim extraction** — the model lists the discrete claims/assertions in it
2. **Technique identification** — for each of the 2 techniques, YES/NO,
   guided by explicit "distinction from X" rules in the codebook (e.g. a
   statement that rejects binary blame is NOT the black-and-white fallacy; a
   short but substantive sentence is NOT a thought-terminating cliché)

This two-step, guardrailed design replaced an earlier flat free-text
extraction prompt after it was observed to both under-flag isolated,
decontextualized span fragments (near-zero recall) and over-flag ordinary
prose in full articles (false positives from surface-pattern matching, e.g.
tagging a mainstream scientific causal claim as "oversimplification").

### Data
- `oversimplification_data.csv` — ~190 SemEval-derived spans labeled
  `Black-and-White_Fallacy` or `Thought-terminating_Cliches` (rows with other
  labels are skipped). No article full-text is included, so `start`/`end`
  offsets aren't resolvable here; eval uses `span_text` directly as a
  standalone "paragraph" — deprioritized for now per the two-step redesign,
  see `src/eval.py`.
- `australia_498sample_climatechange.csv` — ~480 full news articles (not
  labeled for propaganda), the current priority target for running the
  extraction pipeline.
- `aus_sample_preprocessing.R` — an R script that splits the climate-change
  corpus into per-sentence rows (a separate, sentence-level preprocessing
  path; the Python pipeline in this repo uses paragraph-level splitting
  instead, see `src/paragraphs.py`).

### Layout
- `src/codebook.py` — technique definitions, "distinction from X" guardrails,
  few-shot examples, and the mapping from `oversimplification_data.csv`
  technique labels to internal keys
- `src/prompting.py` — two-step prompt construction and JSON-output
  parsing/validation
- `src/paragraphs.py` — splits article full-text into paragraph-sized chunks
- `src/eval.py` — loads `oversimplification_data.csv` and evaluates the
  pipeline against it
- `notebooks/colab_propaganda_poc.ipynb` — the runnable Colab notebook (model
  download, two-step pipeline, evaluation, and a per-paragraph run over the
  climate-change corpus)

### Running the PoC
Open `notebooks/colab_propaganda_poc.ipynb` in Google Colab (`Runtime > Change
runtime type > T4 GPU`) and run the cells top to bottom. It clones this repo
(must be public, or you handle auth yourself), downloads a public GGUF
conversion of Llama-3.1-8B-Instruct (no Hugging Face token required), and runs
the two-step extraction pipeline.

### Known limitations
- Currently scoped to 2 techniques by request (`black_and_white`,
  `thought_terminating_cliche`); `causal_oversimplification` and
  `reductio_ad_hitlerum` were previously covered and can be re-added to
  `src/codebook.py` if needed later.
- `oversimplification_data.csv` has no non-propaganda ("none") examples, so
  eval there only measures recall/technique-confusion, not false-positive
  rate — full-article false positives need to be checked by reading section 8
  output in the notebook.
- An 8B quantized model is less reliable than a larger hosted model at strict
  JSON formatting and nuanced technique judgments.
- Paragraph splitting is a simple heuristic and may over/under-segment.
