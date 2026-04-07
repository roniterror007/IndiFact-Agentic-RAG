from __future__ import annotations

import time
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.personas import build_default_personas
from src.retrieval.indexer import IndicHybridIndexer


class AgentState(TypedDict):
    query: str
    documents: List[Dict[str, Any]]
    draft_answer: Dict[str, Any]
    critic_feedback: Dict[str, Any]
    loop_count: int
    history: List[Dict[str, Any]]
    metrics: Dict[str, Any]


def build_graph(indexer: IndicHybridIndexer):
    personas = build_default_personas(indexer.hybrid_search)
    searcher = personas["searcher"]
    analyst = personas["analyst"]
    critic = personas["critic"]

    def searcher_node(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = searcher.run(query=state["query"], k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        docs = result["documents"]
        metrics = dict(state.get("metrics", {}))
        metrics["searcher_calls"] = int(metrics.get("searcher_calls", 0)) + 1
        metrics["searcher_latency_ms"] = float(metrics.get("searcher_latency_ms", 0.0)) + elapsed_ms
        history = state.get("history", []) + [
            {
                "agent": "Searcher",
                "loop": state["loop_count"],
                "retrieval_count": len(docs),
                "documents": docs,
                "latency_ms": round(elapsed_ms, 2),
            }
        ]
        return {"documents": docs, "history": history, "metrics": metrics}

    def analyst_node(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        draft = analyst.run(query=state["query"], documents=state.get("documents", []))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics = dict(state.get("metrics", {}))
        metrics["analyst_calls"] = int(metrics.get("analyst_calls", 0)) + 1
        metrics["analyst_latency_ms"] = float(metrics.get("analyst_latency_ms", 0.0)) + elapsed_ms
        history = state.get("history", []) + [
            {
                "agent": "Analyst",
                "loop": state["loop_count"],
                "draft_answer": draft,
                "latency_ms": round(elapsed_ms, 2),
            }
        ]
        return {"draft_answer": draft, "history": history, "metrics": metrics}

    def critic_node(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        feedback = critic.run(
            query=state["query"],
            analyst_output=state.get("draft_answer", {}),
            documents=state.get("documents", []),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics = dict(state.get("metrics", {}))
        metrics["critic_calls"] = int(metrics.get("critic_calls", 0)) + 1
        metrics["critic_latency_ms"] = float(metrics.get("critic_latency_ms", 0.0)) + elapsed_ms

        next_loop_count = state["loop_count"]
        if feedback.get("verdict") == "REJECT":
            next_loop_count += 1

        history = state.get("history", []) + [
            {
                "agent": "Critic",
                "loop": state["loop_count"],
                "critic_feedback": feedback,
                "next_loop_count": next_loop_count,
                "latency_ms": round(elapsed_ms, 2),
            }
        ]

        return {
            "critic_feedback": feedback,
            "loop_count": next_loop_count,
            "history": history,
            "metrics": metrics,
        }

    def cannot_verify_node(state: AgentState) -> Dict[str, Any]:
        failed_checks = [
            c.get("reason", "")
            for c in state.get("critic_feedback", {}).get("concept_bottleneck", [])
            if not c.get("pass", False)
        ]

        final_answer = {
            "status": "CANNOT_VERIFY",
            "answer": "Cannot verify after 3 critic-rejected refinement loops.",
            "evidence": [],
            "missing_info": failed_checks or ["Insufficient consistent evidence from retrieved multilingual corpus."],
            "contradictions": ["Critic bottleneck repeatedly rejected the draft."],
        }
        history = state.get("history", []) + [
            {
                "agent": "System",
                "loop": state["loop_count"],
                "event": "Cannot Verify cutoff reached",
            }
        ]
        return {"draft_answer": final_answer, "history": history}

    def finalize_node(state: AgentState) -> Dict[str, Any]:
        history = state.get("history", []) + [
            {
                "agent": "System",
                "loop": state["loop_count"],
                "event": "Finalized",
            }
        ]
        return {"history": history}

    def route_after_critic(state: AgentState) -> str:
        verdict = state.get("critic_feedback", {}).get("verdict", "REJECT")
        if verdict == "ACCEPT":
            return "finalize"
        if state.get("loop_count", 0) >= 3:
            return "cannot_verify"
        return "searcher"

    graph = StateGraph(AgentState)

    graph.add_node("searcher", searcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("cannot_verify", cannot_verify_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "searcher")
    graph.add_edge("searcher", "analyst")
    graph.add_edge("analyst", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "searcher": "searcher",
            "cannot_verify": "cannot_verify",
            "finalize": "finalize",
        },
    )

    graph.add_edge("cannot_verify", END)
    graph.add_edge("finalize", END)

    return graph.compile()


def run_query(indexer: IndicHybridIndexer, query: str) -> Dict[str, Any]:
    app = build_graph(indexer)
    initial_state: AgentState = {
        "query": query,
        "documents": [],
        "draft_answer": {},
        "critic_feedback": {},
        "loop_count": 0,
        "history": [],
        "metrics": {},
    }
    return app.invoke(initial_state)
