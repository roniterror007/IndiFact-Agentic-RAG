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


@dataclass
class OllamaLLM:
    """Minimal Ollama adapter that supports local quantized models."""

    model: str
    api_base: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    timeout_seconds: int = 60
    options: Dict[str, Any] | None = None

    def invoke(self, prompt: str) -> str:
        base = self.api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        url = f"{base}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.options:
            payload["options"].update(self.options)

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
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
                    "missing_info": [f"Ollama call failed: {exc}"],
                    "contradictions": [],
                }
            )

        try:
            parsed = cast(Dict[str, Any], json.loads(body))
            response = parsed.get("response", "")
            if isinstance(response, str) and response.strip():
                return response
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return json.dumps(
            {
                "status": "KNOWLEDGE_GAP",
                "answer": "",
                "evidence": [],
                "missing_info": ["Ollama returned an empty or invalid response."],
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
    confidence_threshold: float = 0.02  # Minimum RRF score to consider relevant

    def run(self, query: str, k: int = 8) -> Dict[str, Any]:
        results = self.retrieval_tool(query, k)
        
        # Filter results by confidence threshold
        high_confidence_results = [
            r for r in results 
            if r.get('rrf_score', 0) >= self.confidence_threshold
        ]
        
        # If we have very low confidence matches, mark as uncertain
        has_low_confidence_only = (
            len(results) > 0 and 
            len(high_confidence_results) == 0 and
            results[0].get('rrf_score', 0) > 0
        )
        
        return {
            "query": query,
            "documents": high_confidence_results if high_confidence_results else results,
            "retrieval_count": len(high_confidence_results) if high_confidence_results else len(results),
            "status": "LOW_CONFIDENCE" if has_low_confidence_only else ("OK" if results else "EMPTY"),
            "confidence_warning": "Results may be tangentially related; corpus may lack specific information." if has_low_confidence_only else None,
        }


@dataclass
class AnalystAgent:
    llm: LLMAdapter

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
        "had", "has", "have", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
        "were", "what", "when", "where", "which", "who", "why", "with", "won", "win",
    }

    _ALIASES = {
        "moon": {"lunar"},
        "lunar": {"moon"},
        "land": {"landed", "landing", "softlanded", "softland", "soft"},
        "landed": {"land", "landing", "softlanded", "softland", "soft"},
        "launch": {"launched", "implemented", "started"},
        "launched": {"launch", "implemented", "started"},
    }

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

    @classmethod
    def _important_tokens(cls, text: str) -> List[str]:
        tokens = re.findall(r"[\w\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F]+", text.lower())
        return [t for t in tokens if len(t) >= 3 and t not in cls._STOPWORDS]

    @classmethod
    def _token_matches(cls, token: str, candidates: set[str]) -> bool:
        if token in candidates:
            return True

        aliases = cls._ALIASES.get(token, set())
        if aliases & candidates:
            return True

        # Lightweight stem/prefix matching to handle land/landed, launch/launched, etc.
        for cand in candidates:
            if len(token) >= 4 and len(cand) >= 4:
                if token.startswith(cand[:4]) or cand.startswith(token[:4]):
                    return True
        return False

    @classmethod
    def _lexical_overlap(cls, query: str, text: str) -> Dict[str, Any]:
        q_tokens = set(cls._important_tokens(query))
        t_tokens = set(cls._important_tokens(text))
        if not q_tokens:
            return {"ratio": 0.0, "shared": 0}

        shared = 0
        for q in q_tokens:
            if cls._token_matches(q, t_tokens):
                shared += 1

        ratio = shared / len(q_tokens)
        return {"ratio": ratio, "shared": shared}

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
            "You are the Analyst agent in a multilingual fact-checking system.\n"
            "Your task: given a query and retrieved evidence, analyze and produce a fact check.\n\n"
            "OUTPUT REQUIREMENTS:\n"
            "Produce ONLY valid JSON matching this exact schema:\n"
            "{\n"
            '  "status": "VERIFIED" | "KNOWLEDGE_GAP",\n'
            '  "answer": "your answer text",\n'
            '  "evidence": ["specific quotes from documents"],\n'
            '  "missing_info": ["any gaps or uncertainties"],\n'
            '  "contradictions": ["any conflicting claims if present"]\n'
            "}\n\n"
            "DECISION RULES:\n"
            '1. Set status to VERIFIED if: the evidence directly answers the query AND you can cite specific text.\n'
            '2. Set status to KNOWLEDGE_GAP only if: evidence is completely absent, irrelevant, or contradictory.\n'
            '3. Do NOT set KNOWLEDGE_GAP just because evidence is partial—use VERIFIED with missing_info notes.\n'
            "4. Even indirect or contextual evidence counts if it addresses the query's core claim.\n\n"
            f"Query:\n{query}\n\n"
            "Retrieved Evidence:\n"
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

        # Deterministic fallback for small/conservative models:
        # if top evidence is strong and lexically aligned with the query,
        # promote KNOWLEDGE_GAP to VERIFIED with a source-grounded answer.
        if (
            parsed["status"] == "KNOWLEDGE_GAP"
            and documents
        ):
            top_doc = documents[0]
            top_text = str(top_doc.get("text", ""))
            top_rrf = float(top_doc.get("rrf_score", 0.0) or 0.0)
            overlap = self._lexical_overlap(query, top_text)

            if top_rrf >= 0.015 and overlap["ratio"] >= 0.35 and overlap["shared"] >= 2:
                parsed["status"] = "VERIFIED"
                if not parsed.get("answer", "").strip():
                    parsed["answer"] = top_text
                if not parsed.get("evidence"):
                    src = str(top_doc.get("source", "unknown"))
                    parsed["evidence"] = [f"source={src} text={top_text[:220]}"]
                parsed["missing_info"] = [
                    (
                        "Status promoted to VERIFIED by deterministic fallback "
                        f"(top_rrf={top_rrf:.4f}, overlap_ratio={overlap['ratio']:.2f})."
                    )
                ]

        return parsed


@dataclass
class CriticAgent:
    llm: LLMAdapter | None = None

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
        return {}

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

    def _llm_concept_bottleneck(
        self,
        query: str,
        answer: str,
        docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]] | None:
        if not self.llm:
            return None
        if os.getenv("CRITIC_USE_LLM", "1").strip().lower() not in {"1", "true", "yes", "on"}:
            return None

        evidence = []
        for i, d in enumerate(docs[:6], start=1):
            evidence.append(
                f"[{i}] source={d.get('source', 'unknown')} lang={d.get('language', 'unknown')} text={d.get('text', '')}"
            )

        prompt = (
            "You are a strict fact-checking critic with concept bottleneck constraints.\n"
            "Evaluate ONLY these concepts: Entity Match, Source Verifiability, Temporal Consistency, Translation Integrity.\n"
            "Return ONLY strict JSON list under key concept_bottleneck.\n"
            "Schema:\n"
            "{\n"
            '  "concept_bottleneck": [\n'
            '    {"name": "Entity Match", "pass": true|false, "score": 0|1, "reason": "..."},\n'
            '    {"name": "Source Verifiability", "pass": true|false, "score": 0|1, "reason": "..."},\n'
            '    {"name": "Temporal Consistency", "pass": true|false, "score": 0|1, "reason": "..."},\n'
            '    {"name": "Translation Integrity", "pass": true|false, "score": 0|1, "reason": "..."}\n'
            "  ]\n"
            "}\n\n"
            f"Query:\n{query}\n\n"
            f"Draft Answer:\n{answer}\n\n"
            "Evidence:\n"
            + "\n".join(evidence)
        )

        parsed = self._extract_json(str(self.llm.invoke(prompt)))
        raw_checks = parsed.get("concept_bottleneck", [])
        if not isinstance(raw_checks, list):
            return None

        normalized: List[Dict[str, Any]] = []
        expected = {
            "Entity Match",
            "Source Verifiability",
            "Temporal Consistency",
            "Translation Integrity",
        }
        for c in raw_checks:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name", "")).strip()
            if name not in expected:
                continue
            normalized.append(
                {
                    "name": name,
                    "pass": bool(c.get("pass", False)),
                    "score": int(c.get("score", 0)),
                    "reason": str(c.get("reason", "LLM critic assessment.")),
                }
            )

        return normalized if len(normalized) == 4 else None

    def run(self, query: str, analyst_output: Dict[str, Any], documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        answer = str(analyst_output.get("answer", ""))
        status = analyst_output.get("status", "KNOWLEDGE_GAP")

        deterministic_checks = [
            self._concept_source_verifiability(documents),
            self._concept_entity_match(query, answer, documents),
            self._concept_temporal_consistency(query, answer, documents),
            self._concept_translation_integrity(query, answer),
        ]

        concept_checks = self._llm_concept_bottleneck(query, answer, documents) or deterministic_checks

        pass_count = sum(1 for c in concept_checks if c["pass"])
        all_pass = pass_count == len(concept_checks)

        verdict = "ACCEPT" if (all_pass and status == "VERIFIED") else "REJECT"
        rationale = "All concept bottlenecks passed." if verdict == "ACCEPT" else "One or more concept bottlenecks failed or analyst flagged knowledge gap."

        return {
            "verdict": verdict,
            "concept_bottleneck": concept_checks,
            "pass_count": pass_count,
            "rationale": rationale,
            "critic_mode": "llm_bottleneck" if concept_checks is not deterministic_checks else "deterministic_bottleneck",
            "next_action": "finalize" if verdict == "ACCEPT" else "retry_search",
        }


def build_default_personas(retrieval_tool: Callable[[str, int], List[Dict[str, Any]]]) -> Dict[str, Any]:
    backend = os.getenv("LLM_BACKEND", "ollama").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    api_base = os.getenv("LLM_API_BASE", "").strip()

    if not model:
        raise RuntimeError(
            "LLM_MODEL is required. Configure a real model (for example, llama3.1:8b on Ollama)."
        )

    if backend in {"openai", "openai_compat"}:
        if not api_key:
            raise RuntimeError("LLM_API_KEY is required for openai_compat backend.")
        llm: LLMAdapter = OpenAICompatLLM(
            model=model,
            api_key=api_key,
            api_base=api_base or "https://api.openai.com/v1",
        )
    elif backend == "ollama":
        critic_options_raw = os.getenv("CRITIC_OLLAMA_OPTIONS", "").strip()
        critic_options: Dict[str, Any] | None = None
        if critic_options_raw:
            try:
                parsed = json.loads(critic_options_raw)
                if isinstance(parsed, dict):
                    critic_options = parsed
            except json.JSONDecodeError:
                critic_options = None
        llm = OllamaLLM(
            model=model,
            api_base=api_base or "http://127.0.0.1:11434",
            options=critic_options,
        )
    else:
        raise RuntimeError(f"Unsupported LLM_BACKEND: {backend}")

    critic_model = os.getenv("CRITIC_MODEL", "").strip()
    critic_api_base = os.getenv("CRITIC_API_BASE", api_base).strip() or "http://127.0.0.1:11434"
    critic_backend = os.getenv("CRITIC_BACKEND", backend).strip().lower()

    critic_llm: LLMAdapter | None = llm
    if critic_model:
        if critic_backend in {"openai", "openai_compat"}:
            if not api_key:
                raise RuntimeError("LLM_API_KEY is required for CRITIC_BACKEND=openai_compat.")
            critic_llm = OpenAICompatLLM(
                model=critic_model,
                api_key=api_key,
                api_base=critic_api_base if critic_api_base else "https://api.openai.com/v1",
            )
        elif critic_backend == "ollama":
            critic_llm = OllamaLLM(model=critic_model, api_base=critic_api_base)

    return {
        "searcher": SearcherAgent(retrieval_tool=retrieval_tool),
        "analyst": AnalystAgent(llm=llm),
        "critic": CriticAgent(llm=critic_llm),
    }
