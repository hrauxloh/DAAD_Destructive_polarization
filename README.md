# DAAD_Destructive_polarization

## Propaganda-technique extraction proof of concept

Identifies language matching 3 techniques from the SemEval / Da San Martino
et al. propaganda taxonomy (causal oversimplification, black-and-white
fallacy, thought-terminating cliché) in unseen news text, using Llama 3.1 8B
Instruct run locally (no paid API) via `llama-cpp-python`, sized to run on a
free-tier Google Colab T4 GPU.

### Data
- `oversimplification_data.csv` — ~390 SemEval-derived spans labeled with one
  of the 3 techniques (`article_id, technique, span_text, start, end`). No
  article full-text is included, so `start`/`end` offsets aren't resolvable
  here; the pipeline and eval use `span_text` directly.
- `australia_498sample_climatechange.csv` — ~480 full news articles (not
  labeled for propaganda), used as the target "unseen text" to run the
  extraction pipeline on.
- `aus_sample_preprocessing.R` — splits the climate-change corpus into
  per-sentence rows.

### Layout
- `src/codebook.py` — technique definitions, few-shot examples, and the
  mapping from `oversimplification_data.csv` technique labels to internal keys
- `src/prompting.py` — prompt construction and JSON-output parsing/validation
- `src/eval.py` — loads `oversimplification_data.csv` and evaluates the
  pipeline against it
- `notebooks/colab_propaganda_poc.ipynb` — the runnable Colab notebook (model
  download, extraction pipeline, evaluation, and a run over the climate-change
  corpus)

### Running the PoC
Open `notebooks/colab_propaganda_poc.ipynb` in Google Colab (`Runtime > Change
runtime type > T4 GPU`) and run the cells top to bottom. It clones this repo,
downloads a public GGUF conversion of Llama-3.1-8B-Instruct (no Hugging Face
token required), and runs the extraction + evaluation pipeline.

### Known limitations
- Only 3 of the taxonomy's techniques are covered (matching the labeled data
  currently in the repo).
- `oversimplification_data.csv` has no non-propaganda ("none") examples, so
  eval here only measures recall/technique-confusion, not false-positive rate.
- An 8B quantized model is less reliable than a larger hosted model at strict
  JSON formatting and nuanced technique judgments. See the notebook's final
  cell for details.
