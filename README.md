# DAAD_Destructive_polarization

## Propaganda-technique extraction proof of concept

Identifies language matching a 7-technique subset of the SemEval / Da San
Martino et al. propaganda taxonomy (name calling, exaggeration/minimization,
causal oversimplification, slogans, black-and-white fallacy, thought-terminating
cliché, reductio ad Hitlerum) in unseen news text, using Llama 3.1 8B Instruct
run locally (no paid API) via `llama-cpp-python`, sized to run on a free-tier
Google Colab T4 GPU.

### Layout
- `src/codebook.py` — technique definitions and few-shot examples
- `src/prompting.py` — prompt construction and JSON-output parsing/validation
- `src/eval.py` — evaluation against the hand-labeled sample set
- `data/sample_labeled_examples.csv` — ~25 hand-labeled example sentences
- `notebooks/colab_propaganda_poc.ipynb` — the runnable Colab notebook (model
  download, extraction pipeline, evaluation)

### Running the PoC
Open `notebooks/colab_propaganda_poc.ipynb` in Google Colab (`Runtime > Change
runtime type > T4 GPU`) and run the cells top to bottom. It clones this repo,
downloads a public GGUF conversion of Llama-3.1-8B-Instruct (no Hugging Face
token required), and runs the extraction + evaluation pipeline.

### Known limitations
This is a small-scale PoC: a hand-written codebook and 25-example labeled set,
and an 8B quantized model, which is less reliable than a larger hosted model at
strict JSON formatting and nuanced technique judgments. See the notebook's
final cell for details.
