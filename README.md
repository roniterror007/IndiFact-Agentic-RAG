# IndiFact-Agentic-RAG

IndiFact-Agentic-RAG is a multilingual fact-checking app built around retrieval-augmented generation and a self-correcting agent loop. It indexes English and Indic-language evidence, retrieves supporting passages with a hybrid dense + sparse search stack, drafts a structured answer, and uses a critic agent to reject unsupported claims and retry up to three times before giving up.

## What It Does

- Indexes custom corpora or a built-in multilingual demo corpus.
- Retrieves evidence with Chroma embeddings plus BM25 rank fusion.
- Runs a LangGraph workflow with Searcher, Analyst, and Critic roles.
- Exposes a Streamlit UI for interactive fact-checking and debug tracing.
- Compares the agentic graph against a naive baseline in an evaluation harness.

## Project Structure

- [run_app.py](run_app.py) - Streamlit launcher.
- [src/ui/app.py](src/ui/app.py) - Main UI.
- [src/orchestration/graph.py](src/orchestration/graph.py) - LangGraph agent flow.
- [src/retrieval/indexer.py](src/retrieval/indexer.py) - Hybrid retrieval and chunking.
- [src/agents/personas.py](src/agents/personas.py) - Searcher, Analyst, and Critic agents.
- [src/evaluation/harness.py](src/evaluation/harness.py) - Baseline vs graph evaluation.
- [src/data/curation.py](src/data/curation.py) - URL scraping and JSONL export.
- [scripts/curate_from_urls.py](scripts/curate_from_urls.py) - CLI wrapper for corpus creation.
- [scripts/run_harness.py](scripts/run_harness.py) - CLI wrapper for evaluation runs.

## Requirements

- Python 3.10 or newer.
- A virtual environment is recommended.
- Optional but recommended for full LLM behavior: Ollama or an OpenAI-compatible API.

Core dependencies are listed in [requirements.txt](requirements.txt). A locked snapshot is available in [requirements.lock.txt](requirements.lock.txt).

## Quick Start

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Start the app.

```powershell
python run_app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Using the App

1. Click **Index Demo Multilingual Corpus** to load a small Hindi, Tamil, Telugu, and English sample set.
2. Or paste a JSON list of documents and click **Index Custom Corpus**.
3. Enter a claim or question in the fact-check box.
4. Run the agentic fact-check and inspect the final verdict.
5. Enable **Debug Mode** to see each graph node update in real time.

The app keeps the indexed corpus and compiled graph in Streamlit session state so you can iterate without restarting the process.

## Corpus Format

Custom documents should be provided as a JSON list of objects with at least a `text` field and an optional `source` field.

```json
[
  {
    "source": "doc-1",
    "text": "Your evidence text here."
  }
]
```

If no source is provided, the app assigns a fallback label. Language metadata is inferred automatically from script detection.

## Data Curation

To scrape a set of pages into a JSONL corpus:

```powershell
python .\scripts\curate_from_urls.py --out datasets/curated_corpus.jsonl <url1> <url2>
```

This writes newline-delimited JSON records in the same `{ "source": "...", "text": "..." }` format used by the app.

## Evaluation Harness

The evaluation harness compares the naive baseline against the agentic graph and writes a report to `datasets/eval_results.json`.

```powershell
python .\scripts\run_harness.py
```

It loads `datasets/sample_corpus.jsonl` and `datasets/sample_eval.jsonl`, then records latency, answer status, and optional Ragas metrics.

## LLM and Ragas Configuration

The app can run with a deterministic placeholder model when no LLM credentials are configured, but better results come from a real model.

Environment variables:

- `LLM_MODEL` - model name for an OpenAI-compatible API or Ollama.
- `LLM_API_KEY` - API key for OpenAI-compatible access.
- `LLM_API_BASE` - API base URL, defaults to the local Ollama-compatible endpoint used by the harness.
- `RAGAS_OLLAMA_LLM_MODEL` - model used by Ragas.
- `RAGAS_OLLAMA_EMBED_MODEL` - embedding model used by Ragas.
- `CRITIC_MODEL_PARAMS` - optional critic model size hint for compute estimates.

If you want to use Ragas locally with Ollama, pull the models first:

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## How The System Works

1. The user submits a claim.
2. The Searcher retrieves evidence with hybrid dense and sparse search.
3. The Analyst turns the evidence into a strict JSON answer.
4. The Critic checks whether the answer is grounded and consistent.
5. If the Critic rejects the answer, the graph loops back to retrieval.
6. After three failed loops, the system returns `CANNOT_VERIFY`.

## Persistent Storage

- Chroma indexes are stored in `./.chroma`.
- Evaluation results are written to `datasets/eval_results.json`.

## Notes

- The repository is tuned for multilingual fact-checking, especially English, Hindi, Tamil, and Telugu.
- The UI includes a demo corpus so you can test the pipeline immediately after installation.
- The graph records per-agent call counts and latency in its final state for inspection in the UI.
