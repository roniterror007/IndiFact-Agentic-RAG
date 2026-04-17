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

The app requires a real LLM configuration. Placeholder fallback has been removed to avoid biased evaluation.

Environment variables:

- `LLM_BACKEND` - `ollama` (default) or `openai_compat`.
- `LLM_MODEL` - required model name used by Analyst.
- `LLM_API_KEY` - required when `LLM_BACKEND=openai_compat`.
- `LLM_API_BASE` - API base URL for selected backend.
- `CRITIC_BACKEND` - optional override for Critic backend (`ollama` or `openai_compat`).
- `CRITIC_MODEL` - optional dedicated Critic model (recommended for local quantized critics).
- `CRITIC_API_BASE` - optional Critic API endpoint override.
- `CRITIC_USE_LLM` - set `1` (default) to run LLM concept-bottleneck checks.
- `CRITIC_OLLAMA_OPTIONS` - optional JSON object of Ollama generation options.
- `RAGAS_OLLAMA_LLM_MODEL` - model used by Ragas.
- `RAGAS_OLLAMA_EMBED_MODEL` - embedding model used by Ragas.
- `CRITIC_MODEL_PARAMS` - optional critic model size hint for compute estimates.

Example (local Ollama with quantized critic):

```powershell
$env:LLM_BACKEND="ollama"
$env:LLM_MODEL="llama3.1:8b"
$env:CRITIC_MODEL="llama3.1:8b-instruct-q4_K_M"
$env:CRITIC_OLLAMA_OPTIONS='{"num_ctx": 2048, "num_predict": 256}'
```

If you want to use Ragas locally with Ollama, pull the models first:

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## How The System Works

1. The user submits a claim.
2. The Searcher retrieves evidence with hybrid dense and sparse search.
   - **Confidence filtering**: Results below RRF score 0.02 flagged as LOW_CONFIDENCE.
   - Low-confidence matches handled gracefully (marked KNOWLEDGE_GAP rather than returned irrelevantly).
3. The Analyst analyzes evidence with improved prompt tuning:
   - Directly answers queries if evidence is present, even if partial.
   - Marks KNOWLEDGE_GAP only when evidence is absent/irrelevant/contradictory.
   - Includes decision rules to increase VERIFIED rate for critic engagement.
4. If Analyst returns `KNOWLEDGE_GAP`, the graph routes directly back to Searcher (max 3 loops).
5. If Analyst returns `VERIFIED`, the Critic runs concept bottleneck checks.
6. If Critic rejects, the graph loops back to retrieval.
7. After three failed loops, the system returns `CANNOT_VERIFY`.

## Recent Improvements

- **Analyst Prompt Tuning** (v2): Enhanced prompt with explicit decision rules to reduce over-conservative KNOWLEDGE_GAP classification. Now uses partial evidence and contextual reasoning.
- **Confidence-Based Filtering**: Searcher detects low-confidence matches (RRF < 0.02) and returns appropriate status flags.
- **Expanded Corpus**: Added 8 multilingual sports documents (FIFA World Cup, Cricket World Cup) for broader query coverage.
- **Graceful Degradation**: Out-of-corpus queries no longer return irrelevant tangential results; system acknowledges information gaps.

## Persistent Storage

- Chroma indexes are stored in `./.chroma`.
- Evaluation results are written to `datasets/eval_results.json`.

## Notes

- The repository is tuned for multilingual fact-checking, especially English, Hindi, Tamil, and Telugu.
- The UI includes a demo corpus so you can test the pipeline immediately after installation.
- The graph records per-agent call counts and latency in its final state for inspection in the UI.
