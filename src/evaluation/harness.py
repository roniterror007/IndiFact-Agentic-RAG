from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.baseline import run_naive_baseline
from src.evaluation.metrics import profile_pipeline_call, to_dict
from src.orchestration.graph import run_query
from src.retrieval.indexer import IndicHybridIndexer


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _status_score(status: str) -> float:
    if status == "VERIFIED":
        return 1.0
    if status == "KNOWLEDGE_GAP":
        return 0.5
    return 0.0


def _safe_get_status(payload: Dict[str, Any]) -> str:
    return str(payload.get("draft_answer", {}).get("status", "UNKNOWN"))


def _maybe_run_ragas(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from ragas import evaluate  # type: ignore
        from ragas.run_config import RunConfig  # type: ignore
        from datasets import Dataset  # type: ignore
        import ragas.metrics as ragas_metrics  # type: ignore
        from ragas.llms import LangchainLLMWrapper  # type: ignore
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
    except ImportError:
        return {
            "enabled": False,
            "reason": "ragas/langchain-ollama dependencies not available in runtime",
        }

    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings  # type: ignore
    except ImportError:
        from langchain_community.chat_models import ChatOllama  # type: ignore
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

    faithfulness = getattr(ragas_metrics, "faithfulness", None) or getattr(ragas_metrics, "_faithfulness", None)
    answer_relevancy = (
        getattr(ragas_metrics, "answer_relevancy", None)
        or getattr(ragas_metrics, "answer_relevance", None)
        or getattr(ragas_metrics, "_answer_relevancy", None)
        or getattr(ragas_metrics, "_answer_relevance", None)
    )

    if faithfulness is None or answer_relevancy is None:
        return {
            "enabled": False,
            "reason": "ragas metrics not available for this installed version",
        }

    ollama_base = os.getenv("LLM_API_BASE", "http://127.0.0.1:11434/v1").strip().rstrip("/")
    if ollama_base.endswith("/v1"):
        ollama_base = ollama_base[:-3]
    ollama_llm_model = os.getenv("RAGAS_OLLAMA_LLM_MODEL", os.getenv("LLM_MODEL", "llama3.1:8b")).strip()
    ollama_embed_model = os.getenv("RAGAS_OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()

    ragas_llm = LangchainLLMWrapper(
        ChatOllama(model=ollama_llm_model, base_url=ollama_base, temperature=0)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=ollama_embed_model, base_url=ollama_base)
    )

    ragas_rows = []
    for row in records:
        ragas_rows.append(
            {
                "question": row["query"],
                "answer": row["graph_answer"],
                "contexts": row.get("contexts", []),
                "ground_truth": row.get("ground_truth", ""),
            }
        )

    if not ragas_rows:
        return {"enabled": False, "reason": "no records"}

    ds = Dataset.from_list(ragas_rows)
    try:
        res = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(timeout=600, max_workers=2),
        )
    except (RuntimeError, ValueError, TypeError, AttributeError, ImportError) as exc:
        return {
            "enabled": False,
            "reason": (
                "ragas evaluation skipped: "
                f"{exc}. Ensure Ollama is running and required models are pulled "
                f"(llm={ollama_llm_model}, embedding={ollama_embed_model})."
            ),
        }

    raw_scores = res.to_pandas().mean(numeric_only=True).to_dict()
    clean_scores: Dict[str, Any] = {}
    for key, value in raw_scores.items():
        if isinstance(value, float) and math.isnan(value):
            clean_scores[key] = None
        else:
            clean_scores[key] = value

    return {"enabled": True, "scores": clean_scores}


def run_evaluation_harness(
    corpus_path: str = "datasets/sample_corpus.jsonl",
    claims_path: str = "datasets/sample_eval.jsonl",
    out_path: str = "datasets/eval_results.json",
) -> Dict[str, Any]:
    corpus = _load_jsonl(Path(corpus_path))
    claims = _load_jsonl(Path(claims_path))

    run_collection = f"indifact_eval_harness_{int(time.time())}"
    idx = IndicHybridIndexer(collection_name=run_collection)
    idx.add_documents(corpus)

    records: List[Dict[str, Any]] = []
    baseline_total = 0.0
    graph_total = 0.0
    baseline_flops_per_byte_total = 0.0
    graph_flops_per_byte_total = 0.0

    for item in claims:
        query = str(item.get("query", "")).strip()
        if not query:
            continue

        baseline, baseline_compute = profile_pipeline_call(
            run_naive_baseline,
            query,
            idx,
            query,
        )
        baseline_ms = baseline_compute.latency_ms

        graph, graph_compute = profile_pipeline_call(
            run_query,
            query,
            idx,
            query,
        )
        graph_ms = graph_compute.latency_ms

        baseline_total += baseline_ms
        graph_total += graph_ms
        baseline_flops_per_byte_total += baseline_compute.flops_per_byte
        graph_flops_per_byte_total += graph_compute.flops_per_byte

        baseline_status = _safe_get_status(baseline)
        graph_status = _safe_get_status(graph)

        contexts = [d.get("text", "") for d in graph.get("documents", [])]
        records.append(
            {
                "query": query,
                "ground_truth": item.get("ground_truth", ""),
                "baseline_status": baseline_status,
                "graph_status": graph_status,
                "baseline_answer": baseline.get("draft_answer", {}).get("answer", ""),
                "graph_answer": graph.get("draft_answer", {}).get("answer", ""),
                "baseline_latency_ms": round(baseline_ms, 2),
                "graph_latency_ms": round(graph_ms, 2),
                "baseline_compute": to_dict(baseline_compute),
                "graph_compute": to_dict(graph_compute),
                "graph_loop_count": graph.get("loop_count", 0),
                "critic_latency_ms": graph.get("metrics", {}).get("critic_latency_ms", 0.0),
                "critic_calls": graph.get("metrics", {}).get("critic_calls", 0),
                "contexts": contexts,
                "baseline_score": _status_score(baseline_status),
                "graph_score": _status_score(graph_status),
            }
        )

    summary = {
        "cases": len(records),
        "baseline_avg_latency_ms": round(baseline_total / max(1, len(records)), 2),
        "graph_avg_latency_ms": round(graph_total / max(1, len(records)), 2),
        "baseline_avg_status_score": round(
            sum(r["baseline_score"] for r in records) / max(1, len(records)), 4
        ),
        "graph_avg_status_score": round(
            sum(r["graph_score"] for r in records) / max(1, len(records)), 4
        ),
        "baseline_avg_flops_per_byte": round(
            baseline_flops_per_byte_total / max(1, len(records)), 4
        ),
        "graph_avg_flops_per_byte": round(
            graph_flops_per_byte_total / max(1, len(records)), 4
        ),
    }

    ragas = _maybe_run_ragas(records)

    payload = {
        "summary": summary,
        "ragas": ragas,
        "records": records,
    }

    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run_evaluation_harness()
    print(json.dumps(result["summary"], indent=2))
