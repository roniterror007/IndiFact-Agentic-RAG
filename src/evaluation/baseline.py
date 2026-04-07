from __future__ import annotations

import time
from typing import Any, Dict

from src.agents.personas import build_default_personas
from src.retrieval.indexer import IndicHybridIndexer


def run_naive_baseline(indexer: IndicHybridIndexer, query: str, k: int = 5) -> Dict[str, Any]:
    """Single-shot baseline: one retrieval pass + one analyst pass, no critic loop."""
    personas = build_default_personas(indexer.hybrid_search)
    searcher = personas["searcher"]
    analyst = personas["analyst"]

    t0 = time.perf_counter()
    retrieval = searcher.run(query=query, k=k)
    draft = analyst.run(query=query, documents=retrieval.get("documents", []))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "query": query,
        "documents": retrieval.get("documents", []),
        "draft_answer": draft,
        "metrics": {
            "latency_ms": round(elapsed_ms, 2),
            "retrieval_count": retrieval.get("retrieval_count", 0),
            "mode": "naive_single_shot",
        },
    }
