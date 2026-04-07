from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    source: str
    language: str


class IndicHybridIndexer:
    """Hybrid retriever (Dense + Sparse) tailored for multilingual Indic corpora."""

    def __init__(
        self,
        collection_name: str = "indifact_chunks",
        persist_directory: str = "./.chroma",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 450,
        chunk_overlap: int = 220,
    ) -> None:
        # Indic scripts often require more overlap to preserve context across chunk boundaries.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory

        # Persist vectors so re-runs don't lose indexed corpus.
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._embedder = SentenceTransformer(embedding_model)

        self._chunks: List[ChunkRecord] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: BM25Okapi | None = None

        self._hydrate_from_collection()

    def _hydrate_from_collection(self) -> None:
        """Rebuild in-memory state from persisted collection contents."""
        raw = self._collection.get(include=["documents", "metadatas"])
        ids = raw.get("ids", []) or []
        docs = raw.get("documents", []) or []
        metas = raw.get("metadatas", []) or []

        records: List[ChunkRecord] = []
        for cid, txt, meta in zip(ids, docs, metas):
            metadata = meta or {}
            records.append(
                ChunkRecord(
                    chunk_id=str(cid),
                    text=str(txt),
                    source=str(metadata.get("source", "unknown")),
                    language=str(metadata.get("language", "unknown")),
                )
            )

        self._chunks = records
        self._tokenized_corpus = [self._tokenize_for_bm25(r.text) for r in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    @staticmethod
    def _detect_script_language(text: str) -> str:
        # Lightweight script heuristic used only for metadata and UI diagnostics.
        if re.search(r"[\u0900-\u097F]", text):
            return "hi"
        if re.search(r"[\u0B80-\u0BFF]", text):
            return "ta"
        if re.search(r"[\u0C00-\u0C7F]", text):
            return "te"
        return "en"

    @staticmethod
    def _sentence_split(text: str) -> List[str]:
        # Includes danda separators used in several Indic writing styles.
        pattern = r"(?<=[\.!?\u0964\u0965])\s+"
        parts = [p.strip() for p in re.split(pattern, text) if p.strip()]
        if not parts:
            return [text.strip()] if text.strip() else []
        return parts

    @staticmethod
    def _tokenize_for_bm25(text: str) -> List[str]:
        tokens = re.findall(r"[\w\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F]+", text.lower())
        return tokens if tokens else text.lower().split()

    def _chunk_text(self, text: str) -> List[str]:
        sentences = self._sentence_split(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current = ""

        for sent in sentences:
            candidate = f"{current} {sent}".strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            # Overlap from previous chunk helps preserve entities and relations.
            overlap_text = current[-self.chunk_overlap :] if current else ""
            current = f"{overlap_text} {sent}".strip()

            if len(current) > self.chunk_size:
                chunks.append(current[: self.chunk_size])
                current = current[self.chunk_size - self.chunk_overlap :]

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]

    def add_documents(self, documents: Sequence[Dict[str, Any]]) -> int:
        """
        Index input documents.

        Input document format:
        {
            "text": "...",
            "source": "optional-source-id"
        }
        """
        new_records: List[ChunkRecord] = []

        for idx, doc in enumerate(documents):
            text = str(doc.get("text", "")).strip()
            if not text:
                continue

            source = str(doc.get("source", f"doc-{idx}"))
            language = self._detect_script_language(text)
            chunks = self._chunk_text(text)

            for chunk in chunks:
                new_records.append(
                    ChunkRecord(
                        chunk_id=str(uuid.uuid4()),
                        text=chunk,
                        source=source,
                        language=language,
                    )
                )

        if not new_records:
            return 0

        embeddings = self._embedder.encode([r.text for r in new_records], normalize_embeddings=True).tolist()

        self._collection.add(
            ids=[r.chunk_id for r in new_records],
            documents=[r.text for r in new_records],
            metadatas=[{"source": r.source, "language": r.language} for r in new_records],
            embeddings=embeddings,
        )

        self._chunks.extend(new_records)
        self._tokenized_corpus = [self._tokenize_for_bm25(r.text) for r in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        return len(new_records)

    def _dense_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self._chunks:
            return []

        query_embedding = self._embedder.encode([query], normalize_embeddings=True).tolist()[0]
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, len(self._chunks)),
            include=["documents", "metadatas", "distances"],
        )

        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        ids = raw.get("ids", [[]])[0]
        if len(ids) != len(docs):
            ids = [f"dense-{i}" for i in range(len(docs))]

        results = []
        for rank, (chunk_id, txt, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": txt,
                    "source": (meta or {}).get("source", "unknown"),
                    "language": (meta or {}).get("language", "unknown"),
                    "dense_rank": rank,
                    "dense_score": 1.0 / (1.0 + float(dist)),
                }
            )
        return results

    def _sparse_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self._bm25 or not self._chunks:
            return []

        tokenized_query = self._tokenize_for_bm25(query)
        bm25_scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            rec = self._chunks[idx]
            results.append(
                {
                    "chunk_id": rec.chunk_id,
                    "text": rec.text,
                    "source": rec.source,
                    "language": rec.language,
                    "sparse_rank": rank,
                    "sparse_score": float(bm25_scores[idx]),
                }
            )
        return results

    @staticmethod
    def _rrf_fuse(
        dense: Sequence[Dict[str, Any]],
        sparse: Sequence[Dict[str, Any]],
        top_k: int,
        k_rrf: int = 60,
    ) -> List[Dict[str, Any]]:
        combined: Dict[str, Dict[str, Any]] = {}

        for item in dense:
            cid = item["chunk_id"]
            combined[cid] = {**item, "rrf_score": 1.0 / (k_rrf + item["dense_rank"])}

        for item in sparse:
            cid = item["chunk_id"]
            if cid not in combined:
                combined[cid] = {**item, "rrf_score": 0.0}
            combined[cid]["rrf_score"] += 1.0 / (k_rrf + item["sparse_rank"])

            for key in ("text", "source", "language", "sparse_rank", "sparse_score"):
                combined[cid][key] = item.get(key, combined[cid].get(key))

        ranked = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
        return ranked[:top_k]

    def hybrid_search(self, query: str, k: int = 5, dense_k: int = 8, sparse_k: int = 8) -> List[Dict[str, Any]]:
        """Return top-k chunks using Dense + Sparse retrieval fused with RRF."""
        dense_results = self._dense_search(query=query, k=dense_k)
        sparse_results = self._sparse_search(query=query, k=sparse_k)
        fused = self._rrf_fuse(dense_results, sparse_results, top_k=k)

        for rank, item in enumerate(fused, start=1):
            item["rank"] = rank

        return fused

    def chunk_count(self) -> int:
        """Return total indexed chunk count for UI and health checks."""
        return len(self._chunks)
