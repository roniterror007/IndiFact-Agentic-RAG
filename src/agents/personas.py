from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import cast
from typing import Any, Callable, Dict, List, Protocol
from urllib import error, request


class LLMAdapter(Protocol):
    """Protocol for LLM invocation."""

    def invoke(self, prompt: str) -> str:
        """Invoke the LLM with a prompt and return a string response."""
        raise NotImplementedError


class PlaceholderLLM:
    """Deterministic fallback. Replace with an actual chat model adapter in production."""

    def invoke(self, prompt: str) -> str:
        query_match = re.search(r"Query:\n(.*?)\n\nEvidence:", prompt, flags=re.S)
        query = query_match.group(1).strip() if query_match else ""

        evidence_texts = re.findall(r"text=(.*)", prompt)
        evidence_texts = [e.strip() for e in evidence_texts if e.strip()]

        if not evidence_texts:
            return json.dumps(
                {
                    "status": "KNOWLEDGE_GAP",
                    "answer": "",
                    "evidence": [],
                    "missing_info": ["No retrieved evidence available."],
                    "contradictions": [],
                }
            )

        query_years = set(re.findall(r"(?:19|20)\d{2}", query))
        evidence_years = set(re.findall(r"(?:19|20)\d{2}", " ".join(evidence_texts)))
        missing_years = sorted(list(query_years - evidence_years))

        if missing_years:
            return json.dumps(
                {
                    "status": "KNOWLEDGE_GAP",
                    "answer": "",
                    "evidence": [],
                    "missing_info": [f"No evidence for year(s): {', '.join(missing_years)}"],
                    "contradictions": [],
                }
            )

        return json.dumps(
            {
                "status": "VERIFIED",
                "answer": evidence_texts[0],
                "evidence": ["[1] Retrieved top-ranked chunk"],
                "missing_info": [],
                "contradictions": [],
            }
        )


@dataclass
class OpenAICompatLLM:
    """Minimal OpenAI-compatible chat adapter using standard library HTTP."""

    model: str
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    timeout_seconds: int = 45

    def invoke(self, prompt: str) -> str:
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON that follows the requested schema.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            return json.dumps(
                {
                    "status": "KNOWLEDGE_GAP",
                    "answer": "",
                    "evidence": [],
                    "missing_info": [f"LLM call failed: {exc}"],
                    "contradictions": [],
                }
            )

        try:
            parsed = cast(Dict[str, Any], json.loads(body))
            choices = cast(List[Dict[str, Any]], parsed.get("choices", []))
            if choices:
                message = cast(Dict[str, Any], choices[0].get("message", {}))
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return json.dumps(
            {
                "status": "KNOWLEDGE_GAP",
                "answer": "",
                "evidence": [],
                "missing_info": ["LLM returned an empty or invalid response."],
                "contradictions": [],
            }
        )


@dataclass
class SearcherAgent:
    retrieval_tool: Callable[[str, int], List[Dict[str, Any]]]

    def run(self, query: str, k: int = 8) -> Dict[str, Any]:
        results = self.retrieval_tool(query, k)
        return {
            "query": query,
            "documents": results,
            "retrieval_count": len(results),
            "status": "OK" if results else "EMPTY",
        }


@dataclass
class AnalystAgent:
    llm: LLMAdapter

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        return {
            "status": "KNOWLEDGE_GAP",
            "answer": "",
            "evidence": [],
            "missing_info": ["Model did not return valid JSON"],
            "contradictions": [],
        }

    def run(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not documents:
            return {
                "status": "KNOWLEDGE_GAP",
                "answer": "",
                "evidence": [],
                "missing_info": ["No retrieved evidence available."],
                "contradictions": [],
            }

        evidence_lines = []
        for i, doc in enumerate(documents, start=1):
            evidence_lines.append(
                f"[{i}] source={doc.get('source', 'unknown')} lang={doc.get('language', 'unknown')} text={doc.get('text', '')}"
            )

        prompt = (
            "You are the Analyst agent in a fact-checking system.\n"
            "Given query and evidence, produce ONLY strict JSON with this schema:\n"
            "{\n"
            '  "status": "VERIFIED" | "KNOWLEDGE_GAP",\n'
            '  "answer": "string",\n'
            '  "evidence": ["citation strings"],\n'
            '  "missing_info": ["string"],\n'
            '  "contradictions": ["string"]\n'
            "}\n"
            "If context is missing or conflicting, set status to KNOWLEDGE_GAP.\n\n"
            f"Query:\n{query}\n\n"
            "Evidence:\n"
            + "\n".join(evidence_lines)
        )

        raw = str(self.llm.invoke(prompt))
        parsed = self._extract_json(raw)

        # Enforce strict shape expected by downstream graph nodes.
        parsed.setdefault("status", "KNOWLEDGE_GAP")
        parsed.setdefault("answer", "")
        parsed.setdefault("evidence", [])
        parsed.setdefault("missing_info", [])
        parsed.setdefault("contradictions", [])

        allowed_status = {"VERIFIED", "KNOWLEDGE_GAP"}
        if parsed["status"] not in allowed_status:
            parsed["status"] = "KNOWLEDGE_GAP"

        return parsed


@dataclass
class CriticAgent:
    llm: LLMAdapter | None = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[\w\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F]+", text.lower())

    def _concept_entity_match(self, query: str, answer: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        query_tokens = {t for t in self._tokenize(query) if len(t) > 3}
        answer_tokens = set(self._tokenize(answer))
        evidence_tokens = set(self._tokenize(" ".join([d.get("text", "") for d in docs])))

        overlap = len(query_tokens & (answer_tokens | evidence_tokens))
        passed = overlap >= max(1, len(query_tokens) // 4) if query_tokens else True

        return {
            "name": "Entity Match",
            "pass": passed,
            "score": overlap,
            "reason": "Core query entities align with answer/evidence." if passed else "Entity overlap is weak; likely unsupported claim.",
        }

    def _concept_source_verifiability(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not docs:
            return {
                "name": "Source Verifiability",
                "pass": False,
                "score": 0,
                "reason": "No retrieved sources available for verification.",
            }

        valid_sources = [d for d in docs if str(d.get("source", "")).strip() and d.get("source") != "unknown"]
        ratio = len(valid_sources) / max(1, len(docs))
        passed = ratio >= 0.7

        return {
            "name": "Source Verifiability",
            "pass": passed,
            "score": round(ratio, 3),
            "reason": "Retrieved evidence has traceable sources." if passed else "Too many chunks have missing/unknown source metadata.",
        }

    def _concept_temporal_consistency(self, query: str, answer: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        years_query = set(re.findall(r"(?:19|20)\d{2}", query))
        years_answer = set(re.findall(r"(?:19|20)\d{2}", answer))
        years_docs = set(re.findall(r"(?:19|20)\d{2}", " ".join([d.get("text", "") for d in docs])))

        if years_query and years_docs and (years_query - years_docs):
            asked = ", ".join(sorted(years_query))
            available = ", ".join(sorted(years_docs))
            return {
                "name": "Temporal Consistency",
                "pass": False,
                "score": 0,
                "reason": f"Year mismatch: query asks {asked}, but retrieved evidence supports {available}.",
            }

        if not years_answer:
            return {
                "name": "Temporal Consistency",
                "pass": True,
                "score": 1,
                "reason": "No explicit year in draft answer; temporal check not triggered.",
            }

        missing = sorted(list(years_answer - years_docs))
        passed = not missing

        return {
            "name": "Temporal Consistency",
            "pass": passed,
            "score": 0 if missing else 1,
            "reason": "Temporal claims are grounded in evidence." if passed else f"Unsupported year(s) in answer: {', '.join(missing)}",
        }

    def _concept_translation_integrity(self, query: str, answer: str) -> Dict[str, Any]:
        query_has_indic = bool(re.search(r"[\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F]", query))
        answer_has_indic = bool(re.search(r"[\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F]", answer))

        # If query is Indic but answer has no Indic script and no transliteration hints, flag a likely translation issue.
        translit_hint = bool(re.search(r"\([^)]+\)", answer))
        passed = (not query_has_indic) or answer_has_indic or translit_hint

        return {
            "name": "Translation Integrity",
            "pass": passed,
            "score": 1 if passed else 0,
            "reason": "Language transfer appears consistent." if passed else "Possible translation drift; answer may lose source-language nuance.",
        }

    def run(self, query: str, analyst_output: Dict[str, Any], documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        answer = str(analyst_output.get("answer", ""))
        status = analyst_output.get("status", "KNOWLEDGE_GAP")

        concept_checks = [
            self._concept_source_verifiability(documents),
            self._concept_entity_match(query, answer, documents),
            self._concept_temporal_consistency(query, answer, documents),
            self._concept_translation_integrity(query, answer),
        ]

        pass_count = sum(1 for c in concept_checks if c["pass"])
        all_pass = pass_count == len(concept_checks)

        verdict = "ACCEPT" if (all_pass and status == "VERIFIED") else "REJECT"
        rationale = "All concept bottlenecks passed." if verdict == "ACCEPT" else "One or more concept bottlenecks failed or analyst flagged knowledge gap."

        return {
            "verdict": verdict,
            "concept_bottleneck": concept_checks,
            "pass_count": pass_count,
            "rationale": rationale,
            "next_action": "finalize" if verdict == "ACCEPT" else "retry_search",
        }


def build_default_personas(retrieval_tool: Callable[[str, int], List[Dict[str, Any]]]) -> Dict[str, Any]:
    model = os.getenv("LLM_MODEL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").strip()

    if model and api_key:
        llm: LLMAdapter = OpenAICompatLLM(
            model=model,
            api_key=api_key,
            api_base=api_base,
        )
    else:
        llm = PlaceholderLLM()

    return {
        "searcher": SearcherAgent(retrieval_tool=retrieval_tool),
        "analyst": AnalystAgent(llm=llm),
        "critic": CriticAgent(llm=llm),
    }
