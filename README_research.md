# Research Workflow Additions

## 1) Git Bootstrap

```powershell
pwsh ./scripts/init_git.ps1
```

## 2) Dataset Curation

Scrape source pages into JSONL corpus:

```powershell
c:/Users/Ronit/Desktop/LLM/.venv/Scripts/python.exe ./scripts/curate_from_urls.py --out datasets/curated_corpus.jsonl <url1> <url2>
```

## 3) Naive Baseline and Agentic Evaluation Harness

Runs baseline vs graph and writes `datasets/eval_results.json`.

```powershell
c:/Users/Ronit/Desktop/LLM/.venv/Scripts/python.exe ./scripts/run_harness.py
```

### Ollama requirements for Ragas in harness

Ragas in this project is wired to Ollama (not OpenAI). Before running the harness:

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Optional overrides:

```powershell
$env:RAGAS_OLLAMA_LLM_MODEL="llama3.1:8b"
$env:RAGAS_OLLAMA_EMBED_MODEL="nomic-embed-text"
```

## 4) Compute Budget Tracking

Per-query loop metrics are attached in graph state under `metrics` and shown in the UI final section.

Tracked fields:
- `searcher_calls`
- `searcher_latency_ms`
- `analyst_calls`
- `analyst_latency_ms`
- `critic_calls`
- `critic_latency_ms`

You can set critic model parameter assumption for FLOPs/byte estimation used in evaluation utilities:

```powershell
$env:CRITIC_MODEL_PARAMS="8000000000"
```

## 5) Dependency Lock

- Floating constraints: `requirements.txt`
- Locked snapshot: `requirements.lock.txt`
