from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.evaluation.harness import run_evaluation_harness
from src.orchestration.graph import build_graph
from src.retrieval.indexer import IndicHybridIndexer


st.set_page_config(page_title="IndiFact-Agentic-RAG", page_icon="\U0001F9E0", layout="wide")
st.title("IndiFact-Agentic-RAG")
st.caption("Self-correcting multilingual fact-checking with cyclic multi-agent refinement")


def _init_indexer() -> IndicHybridIndexer:
    if "indexer" not in st.session_state:
        st.session_state.indexer = IndicHybridIndexer()
    return st.session_state.indexer


def _init_graph(idx: IndicHybridIndexer):
    if "compiled_graph" not in st.session_state:
        st.session_state.compiled_graph = build_graph(idx)
    return st.session_state.compiled_graph


def _render_documents(docs: List[Dict[str, Any]]) -> None:
    if not docs:
        st.info("No chunks retrieved in this step.")
        return

    for d in docs:
        with st.container(border=True):
            st.markdown(f"**Rank #{d.get('rank', '?')} | Source:** {d.get('source', 'unknown')} | **Lang:** {d.get('language', 'unknown')}")
            st.write(d.get("text", ""))
            st.caption(
                f"RRF={d.get('rrf_score', 0):.4f}, dense_rank={d.get('dense_rank', '-')}, sparse_rank={d.get('sparse_rank', '-')}"
            )


def _seed_demo_corpus(idx: IndicHybridIndexer) -> None:
    docs = [
        {
            "source": "demo-en-1",
            "text": "The Chandrayaan-3 mission by ISRO successfully soft-landed near the lunar south pole on 23 August 2023.",
        },
        {
            "source": "demo-hi-1",
            "text": "चंद्रयान-3 ने 23 अगस्त 2023 को चंद्रमा के दक्षिणी ध्रुव के पास सफल सॉफ्ट लैंडिंग की।",
        },
        {
            "source": "demo-ta-1",
            "text": "சந்திரயான்-3 23 ஆகஸ்ட் 2023 அன்று நிலவின் தெற்கு துருவம் அருகே வெற்றிகரமாக தரையிறங்கியது.",
        },
        {
            "source": "demo-te-1",
            "text": "చంద్రయాన్-3 23 ఆగస్టు 2023న చంద్రుడి దక్షిణ ధ్రువం సమీపంలో విజయవంతంగా ల్యాండైంది.",
        },
    ]
    demo_chunk_count = idx.add_documents(docs)
    st.success(f"Demo corpus indexed with {demo_chunk_count} chunks.")


indexer = _init_indexer()
compiled_graph = _init_graph(indexer)

with st.sidebar:
    st.subheader("Corpus")
    st.metric("Indexed Chunks", indexer.chunk_count())
    debug_mode = st.toggle(
        "Debug Mode",
        value=False,
        help="Show per-node interaction stream and full internal state.",
    )
    if st.button("Index Demo Multilingual Corpus"):
        _seed_demo_corpus(indexer)
        st.session_state.compiled_graph = build_graph(indexer)
        st.session_state.compiled_graph = st.session_state.compiled_graph

    raw_corpus = st.text_area(
        "Add custom corpus as JSON list",
        value='[{"source": "my-doc-1", "text": "Your English/Hindi/Tamil/Telugu evidence here."}]',
        height=160,
    )

    if st.button("Index Custom Corpus"):
        try:
            parsed = json.loads(raw_corpus)
            if not isinstance(parsed, list):
                raise ValueError("JSON must be a list of documents.")
            custom_chunk_count = indexer.add_documents(parsed)
            st.success(f"Indexed {custom_chunk_count} chunks.")
            st.session_state.compiled_graph = build_graph(indexer)
            st.session_state.compiled_graph = st.session_state.compiled_graph
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Failed to index corpus: {e}")

    st.subheader("Evaluation")
    if st.button("Run Baseline vs Graph Harness"):
        try:
            st.session_state.eval_result = run_evaluation_harness(
                corpus_path="datasets/sample_corpus.jsonl",
                claims_path="datasets/sample_eval.jsonl",
                out_path="datasets/eval_results.json",
            )
            st.success("Evaluation harness completed.")
        except (RuntimeError, ValueError, TypeError, FileNotFoundError) as e:
            st.error(f"Evaluation failed: {e}")

    if st.button("Load Last Evaluation Report"):
        report_path = Path("datasets/eval_results.json")
        if report_path.exists():
            st.session_state.eval_result = json.loads(report_path.read_text(encoding="utf-8"))
            st.success("Loaded datasets/eval_results.json")
        else:
            st.warning("No saved evaluation report found yet.")

if "eval_result" in st.session_state:
    eval_result = st.session_state.eval_result
    with st.expander("Evaluation Dashboard", expanded=False):
        summary = eval_result.get("summary", {})
        ragas = eval_result.get("ragas", {})
        records = eval_result.get("records", [])

        st.markdown("### Evaluation Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cases", summary.get("cases", 0))
        c2.metric("Baseline Avg Latency (ms)", summary.get("baseline_avg_latency_ms", 0.0))
        c3.metric("Graph Avg Latency (ms)", summary.get("graph_avg_latency_ms", 0.0))
        c4, c5 = st.columns(2)
        c4.metric("Baseline Status Score", summary.get("baseline_avg_status_score", 0.0))
        c5.metric("Graph Status Score", summary.get("graph_avg_status_score", 0.0))

        st.markdown("### Ragas")
        st.json(ragas)

        st.markdown("### Per-Query Records")
        st.json(records)

st.subheader("Fact-Check Query")
query = st.text_input("Enter a claim or question", placeholder="Example: Did Chandrayaan-3 land in 2023?")
run_btn = st.button("Run Agentic Fact-Check", type="primary")

if run_btn:
    if indexer.chunk_count() == 0:
        st.warning("No corpus is indexed yet. Click 'Index Demo Multilingual Corpus' or add custom corpus first.")
    elif not query.strip():
        st.warning("Please enter a query.")
    else:
        initial_state = {
            "query": query.strip(),
            "documents": [],
            "draft_answer": {},
            "critic_feedback": {},
            "loop_count": 0,
            "history": [],
            "metrics": {},
        }

        final_state = None

        if debug_mode:
            timeline = st.container()
            with timeline:
                st.markdown("### Agent Interaction Stream")
                for event in compiled_graph.stream(initial_state, stream_mode="updates"):
                    for node_name, update in event.items():
                        with st.expander(f"Node: {node_name}", expanded=True):
                            if node_name == "searcher":
                                st.markdown(f"**Loop:** {update.get('history', [{}])[-1].get('loop', '?')}")
                                st.markdown("**Retrieved Chunks:**")
                                _render_documents(update.get("documents", []))

                            elif node_name == "analyst":
                                st.markdown("**Analyst Draft (Strict JSON):**")
                                st.json(update.get("draft_answer", {}))

                            elif node_name == "critic":
                                feedback = update.get("critic_feedback", {})
                                st.markdown(f"**Critic Verdict:** {feedback.get('verdict', 'UNKNOWN')}")
                                st.markdown(f"**Reasoning:** {feedback.get('rationale', '')}")
                                st.markdown("**Concept Bottleneck:**")
                                for c in feedback.get("concept_bottleneck", []):
                                    st.write(
                                        f"- {c.get('name')}: {'PASS' if c.get('pass') else 'FAIL'} | {c.get('reason')}"
                                    )
                                st.markdown(f"**Next Loop Count:** {update.get('loop_count', '?')}")

                            elif node_name == "cannot_verify":
                                st.error("Reached maximum retry limit (3). System cannot verify the claim.")
                                st.json(update.get("draft_answer", {}))

                            elif node_name == "finalize":
                                st.success("Draft accepted by Critic. Finalizing answer.")

            final_state = compiled_graph.invoke(initial_state)
        else:
            final_state = compiled_graph.invoke(initial_state)

        draft_answer = final_state.get("draft_answer", {})
        critic_feedback = final_state.get("critic_feedback", {})
        status = draft_answer.get("status", "UNKNOWN")
        answer = draft_answer.get("answer", "")
        verdict = critic_feedback.get("verdict", "UNKNOWN")

        st.markdown("### Final Verdict")
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", status)
        col2.metric("Critic", verdict)
        col3.metric("Loops", final_state.get("loop_count", 0))

        if status in {"VERIFIED"}:
            st.success(answer or "Claim verified with available evidence.")
        elif status in {"CANNOT_VERIFY", "KNOWLEDGE_GAP"}:
            st.warning(answer or "Could not verify with current indexed corpus.")
        else:
            st.info(answer or "Completed with non-standard status.")

        evidence = draft_answer.get("evidence", [])
        if evidence:
            st.markdown("### Top Evidence")
            for ev in evidence:
                st.write(f"- {ev}")

        metrics = final_state.get("metrics", {})
        if metrics:
            st.markdown("### Compute Budget")
            st.write(
                f"- Searcher: {metrics.get('searcher_calls', 0)} call(s), {metrics.get('searcher_latency_ms', 0.0):.2f} ms total"
            )
            st.write(
                f"- Analyst: {metrics.get('analyst_calls', 0)} call(s), {metrics.get('analyst_latency_ms', 0.0):.2f} ms total"
            )
            st.write(
                f"- Critic: {metrics.get('critic_calls', 0)} call(s), {metrics.get('critic_latency_ms', 0.0):.2f} ms total"
            )

        if debug_mode:
            st.markdown("### Final Output")
            st.json(draft_answer)
            st.markdown("### Full State")
            st.json(final_state)
