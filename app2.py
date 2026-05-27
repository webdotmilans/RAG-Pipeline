# -*- coding: utf-8 -*-
"""
Production-grade RAG pipeline with 10 hardened layers:
1.  Ingest + normalize     - hash, version, mtime, trust, dedup
2.  Hybrid retrieval       - BM25 + dense + RRF + query routing
3.  Two-stage reranking    - ANN (Chroma HNSW) + cross-encoder
4.  Source confidence      - freshness * trust * retrieval consistency
5.  Constraint generation  - context-only, no external knowledge
6.  Citation-backed answer - per-sentence grounding with doc+page+ts
7.  Hallucination fallback - threshold guard + insufficient-evidence
8.  Continuous evals       - adversarial + recall + hallucination rate
9.  Cache + memory layer   - LRU caches for queries, embeddings, LLM
10. Observability          - structured traces, metrics, dashboards
"""
import streamlit as st
import os
import httpx
import numpy as np
import chromadb
import hashlib
import json
import math
import re
import time
import threading
import fitz  # PyMuPDF -- lightweight, no langchain needed
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

_http = httpx.Client(verify=False, timeout=120)


# ============================================================
# LAYER 10: OBSERVABILITY
# ============================================================
class TraceLogger:
    """Captures every pipeline step with timing + payload for auditability."""

    def __init__(self, max_events: int = 200):
        self.max_events = max_events
        self.events: List[Dict] = []
        self.t0: float = time.time()
        self._lock = threading.Lock()

    def log(self, stage: str, **payload):
        with self._lock:
            self.events.append({
                "ts": round(time.time() - self.t0, 4),
                "stage": stage,
                "payload": payload,
            })
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events :]

    def reset(self):
        self.events = []
        self.t0 = time.time()

    def to_json(self) -> str:
        return json.dumps(self.events, indent=2, default=str)

    def stage_durations(self) -> Dict[str, float]:
        durations: Dict[str, float] = defaultdict(float)
        last_ts = 0.0
        for ev in self.events:
            durations[ev["stage"]] += round(ev["ts"] - last_ts, 4)
            last_ts = ev["ts"]
        return dict(durations)


class MetricsCollector:
    """Aggregate metrics: latency, hallucination rate, recall, cache hits."""

    def __init__(self):
        self.queries_total: int = 0
        self.queries_grounded: int = 0
        self.queries_fallback: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.latencies_ms: List[float] = []
        self.confidence_scores: List[float] = []

    def record_query(self, latency_ms: float, confidence: float, grounded: bool, fallback: bool):
        self.queries_total += 1
        self.latencies_ms.append(latency_ms)
        self.confidence_scores.append(confidence)
        if grounded:
            self.queries_grounded += 1
        if fallback:
            self.queries_fallback += 1
        if len(self.latencies_ms) > 1000:
            self.latencies_ms = self.latencies_ms[-1000 :]
            self.confidence_scores = self.confidence_scores[-1000 :]

    def record_cache(self, hit: bool):
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def summary(self) -> Dict[str, Any]:
        latencies = self.latencies_ms or [0.0]
        return {
            "queries_total": self.queries_total,
            "grounded_pct": round(100 * self.queries_grounded / max(self.queries_total, 1), 2),
            "fallback_pct": round(100 * self.queries_fallback / max(self.queries_total, 1), 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 1),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1),
            "avg_confidence": round(float(np.mean(self.confidence_scores or [0])), 3),
            "cache_hit_rate": round(self.cache_hits / max(self.cache_hits + self.cache_misses, 1), 3),
        }


# ============================================================
# LAYER 9: CACHE + MEMORY LAYER
# ============================================================
class TTLCache:
    """Thread-safe LRU cache with time-to-live."""

    def __init__(self, max_size: int = 256, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._store: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                return None
            ts, val = self._store[key]
            if time.time() - ts > self.ttl:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return val

    def put(self, key: str, value: Any):
        with self._lock:
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self):
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


def _cache_key(*parts) -> str:
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ============================================================
# LIGHTWEIGHT API WRAPPERS (no langchain, no torch)
# ============================================================
class GroqLLM:
    """Direct Groq API client via httpx. Includes retry + LRU cache + trace hooks."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        cache: Optional[TTLCache] = None,
        tracer: Optional[TraceLogger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.tracer = tracer
        self.metrics = metrics

    def invoke(self, prompt: str, max_tokens: Optional[int] = None, cacheable: bool = True) -> str:
        tokens = max_tokens or self.max_tokens
        key = _cache_key("llm", self.model, self.temperature, tokens, prompt)
        if cacheable and self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                if self.metrics:
                    self.metrics.record_cache(hit=True)
                if self.tracer:
                    self.tracer.log("llm.cache_hit", model=self.model, prompt_chars=len(prompt))
                return cached
            if self.metrics:
                self.metrics.record_cache(hit=False)

        t0 = time.time()
        for attempt in range(6):
            try:
                resp = _http.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": tokens,
                    },
                )
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("retry-after", 2 ** attempt * 5))
                    if self.tracer:
                        self.tracer.log("llm.rate_limit", attempt=attempt, retry_after=retry_after)
                    time.sleep(min(retry_after, 60))
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                if cacheable and self.cache is not None:
                    self.cache.put(key, content)
                if self.tracer:
                    self.tracer.log(
                        "llm.invoke",
                        model=self.model,
                        prompt_chars=len(prompt),
                        output_chars=len(content),
                        latency_ms=round((time.time() - t0) * 1000, 1),
                    )
                return content
            except Exception as e:
                if attempt == 5:
                    if self.tracer:
                        self.tracer.log("llm.failed", error=str(e)[:200])
                    return "[Rate limit reached -- please wait 30 seconds and try again]"
                time.sleep(min(2 ** attempt * 3, 45))
        return "[LLM call failed after retries]"


class HFEmbeddings:
    """HuggingFace Inference API embeddings via huggingface_hub (lightweight, no torch)."""

    def __init__(self, api_key: str, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(model=model, token=api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        result = self.client.feature_extraction(texts)
        if hasattr(result, 'tolist'):
            return result.tolist()
        return result


# ============================================================
# LAYER 3: CROSS-ENCODER RERANKER (two-stage retrieval, stage 2)
# ============================================================
class CrossEncoderReranker:
    """
    Cross-encoder reranker via HuggingFace Inference API.

    Stage 1: ANN retrieval returns ~50 candidates (cheap).
    Stage 2: Cross-encoder scores each (query, passage) pair (deep, expensive).
    Falls back to LLM rerank when API unreachable.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        cache: Optional[TTLCache] = None,
        tracer: Optional[TraceLogger] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.cache = cache
        self.tracer = tracer
        self.url = f"https://api-inference.huggingface.co/models/{model}"

    def score_pairs(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        cache_key = _cache_key("ce", self.model, query, hashlib.md5("|".join(passages).encode()).hexdigest())
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                if self.tracer:
                    self.tracer.log("rerank.cache_hit", n=len(passages))
                return cached

        t0 = time.time()
        try:
            payload = {"inputs": {"source_sentence": query, "sentences": passages}}
            resp = _http.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                scores = resp.json()
                if isinstance(scores, list) and len(scores) == len(passages):
                    if self.cache:
                        self.cache.put(cache_key, scores)
                    if self.tracer:
                        self.tracer.log(
                            "rerank.cross_encoder",
                            n=len(passages),
                            latency_ms=round((time.time() - t0) * 1000, 1),
                        )
                    return [float(s) for s in scores]
        except Exception as e:
            if self.tracer:
                self.tracer.log("rerank.cross_encoder_failed", error=str(e)[:200])
        return []  # caller should fall back to LLM rerank


# ============================================================
# 1. EMBEDDING MANAGER (with cache + trace + per-vector LRU)
# ============================================================
class EmbeddingManager:
    BATCH_SIZE = 32

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache: Optional[TTLCache] = None,
        tracer: Optional[TraceLogger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.model_name = model_name
        self.client = HFEmbeddings(
            api_key=os.getenv("HF_API_TOKEN"),
            model=model_name,
        )
        self.cache = cache
        self.tracer = tracer
        self.metrics = metrics

    def _cached_embed(self, texts: List[str]) -> List[List[float]]:
        if self.cache is None:
            return self.client.embed(texts)
        results: List[Optional[List[float]]] = [None] * len(texts)
        misses_idx: List[int] = []
        misses_text: List[str] = []
        for i, t in enumerate(texts):
            key = _cache_key("emb", self.model_name, t)
            cached = self.cache.get(key)
            if cached is not None:
                results[i] = cached
                if self.metrics:
                    self.metrics.record_cache(hit=True)
            else:
                misses_idx.append(i)
                misses_text.append(t)
                if self.metrics:
                    self.metrics.record_cache(hit=False)
        if misses_text:
            new_embs = self.client.embed(misses_text)
            for j, idx in enumerate(misses_idx):
                results[idx] = new_embs[j]
                self.cache.put(_cache_key("emb", self.model_name, texts[idx]), new_embs[j])
        return [r for r in results if r is not None]

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        t0 = time.time()
        all_emb: list = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            all_emb.extend(self._cached_embed(batch))
        arr = np.array(all_emb)
        if self.tracer:
            self.tracer.log(
                "embed.batch",
                n=len(texts),
                model=self.model_name,
                latency_ms=round((time.time() - t0) * 1000, 1),
            )
        return arr


# ============================================================
# 2. BM25 INDEX (lightweight Okapi BM25 for keyword search)
# ============================================================
class BM25Index:

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = 0
        self.avgdl = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []
        self.doc_lens: List[int] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def build(self, documents: List[str]):
        self.N = len(documents)
        self.term_freqs, self.doc_lens, self.doc_freqs = [], [], {}
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lens.append(len(tokens))
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for t in tf:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
        self.avgdl = sum(self.doc_lens) / max(self.N, 1)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        qtokens = self._tokenize(query)
        scores = []
        for i in range(self.N):
            s = 0.0
            dl = self.doc_lens[i]
            for t in qtokens:
                df = self.doc_freqs.get(t, 0)
                if df == 0:
                    continue
                idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                tf = self.term_freqs[i].get(t, 0)
                s += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================
# 3. VECTOR STORE (ChromaDB persistent, HNSW-cosine, scale-ready)
# ============================================================
class VectorStore:
    """
    ChromaDB PersistentClient with HNSW cosine index.

    For 10M+ docs: switch to a managed vector DB (Qdrant, Weaviate, Pinecone)
    by replacing this class. The HybridRetriever interface is stable.
    """

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_dir: str = "./chroma_db",
        reset: bool = True,
        tracer: Optional[TraceLogger] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.tracer = tracer
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        if reset:
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 256,  # higher = better recall during build
                "hnsw:search_ef": 128,  # higher = better recall during query
                "hnsw:M": 32,  # graph degree -- higher = better recall, more memory
            },
        )

    @staticmethod
    def _sanitize_meta(meta: Dict) -> Dict:
        return {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in meta.items()
            if v is not None
        }

    def add(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict]):
        t0 = time.time()
        ids = [f"doc_{i}" for i in range(len(texts))]
        clean_metas = [self._sanitize_meta(m) for m in metadatas]
        batch = 5000
        for j in range(0, len(ids), batch):
            s = slice(j, j + batch)
            self.collection.add(
                ids=ids[s],
                embeddings=[e.tolist() for e in embeddings[s]],
                metadatas=clean_metas[s],
                documents=texts[s],
            )
        if self.tracer:
            self.tracer.log(
                "vectorstore.add",
                n=len(texts),
                latency_ms=round((time.time() - t0) * 1000, 1),
            )

    def query_vectors(self, embedding: np.ndarray, n_results: int = 20):
        t0 = time.time()
        result = self.collection.query(
            query_embeddings=[embedding.tolist()], n_results=n_results
        )
        if self.tracer:
            self.tracer.log(
                "vectorstore.query",
                n_results=n_results,
                latency_ms=round((time.time() - t0) * 1000, 1),
            )
        return result

    def count(self):
        return self.collection.count()


# ============================================================
# 4. HYBRID RETRIEVER (Dense + BM25 + RRF + Dedup + Consistency)
# ============================================================
class HybridRetriever:
    """
    Hybrid retrieval combining dense vector search and sparse BM25 with RRF fusion.

    Returns retrieval consistency: how many of the N query variants returned
    each chunk (a strong signal of relevance, used by ConfidenceScorer).
    """

    RRF_K = 60

    def __init__(
        self,
        vector_store: VectorStore,
        emb_manager: EmbeddingManager,
        bm25: BM25Index,
        texts: List[str],
        metas: List[Dict],
        tracer: Optional[TraceLogger] = None,
    ):
        self.vs = vector_store
        self.emb = emb_manager
        self.bm25 = bm25
        self.texts = texts
        self.metas = metas
        self.tracer = tracer

    @staticmethod
    def detect_query_type(query: str) -> str:
        """Route queries: 'factual' (numbers/names) → BM25-heavy; 'conceptual' → dense-heavy."""
        q = query.lower()
        factual_signals = [
            r"\bhow many\b", r"\bhow much\b", r"\bwhat (is|are) the (number|score|percent)",
            r"\bwhen\b", r"\bwho\b", r"\bwhich\b",
            r"\b\d+%\b", r"\b\d{4}\b",
        ]
        for pat in factual_signals:
            if re.search(pat, q):
                return "factual"
        return "conceptual"

    def _rrf(self, rankings: List[List[int]]) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.RRF_K + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _dedup(self, docs: List[Dict], threshold: float = 0.85) -> List[Dict]:
        if not docs:
            return docs
        unique = [docs[0]]
        unique_words = [set(docs[0]["content"].lower().split())]
        for d in docs[1:]:
            d_words = set(d["content"].lower().split())
            is_dup = False
            for uw in unique_words:
                if d_words and uw:
                    overlap = len(d_words & uw) / min(len(d_words), len(uw))
                    if overlap > threshold:
                        is_dup = True
                        break
            if not is_dup:
                unique.append(d)
                unique_words.append(d_words)
        return unique

    def retrieve(self, queries: List[str], top_k: int = 5) -> List[Dict]:
        t0 = time.time()
        num_docs = len(self.texts)
        fetch_k = min(max(top_k * 6, 40), num_docs)
        all_rankings: List[List[int]] = []
        consistency: Dict[int, int] = defaultdict(int)
        seen_per_query: List[set] = []

        # Adaptive weighting: weight BM25 more for factual queries
        primary_query_type = self.detect_query_type(queries[0]) if queries else "conceptual"
        bm25_repeats = 2 if primary_query_type == "factual" else 1

        for q in queries:
            seen: set = set()
            # Dense vector search
            q_emb = self.emb.generate_embeddings([q])[0]
            vr = self.vs.query_vectors(q_emb, n_results=fetch_k)
            if vr["ids"] and vr["ids"][0]:
                ranking = [int(did.split("_")[1]) for did in vr["ids"][0]]
                all_rankings.append(ranking)
                seen.update(ranking)

            # Sparse BM25 keyword search (repeated for factual queries → higher weight)
            bm25_hits = self.bm25.search(q, top_k=fetch_k)
            if bm25_hits:
                bm25_ranking = [idx for idx, _ in bm25_hits]
                for _ in range(bm25_repeats):
                    all_rankings.append(bm25_ranking)
                seen.update(bm25_ranking)

            seen_per_query.append(seen)

        # Retrieval consistency: how many of the N queries surfaced each chunk
        for s in seen_per_query:
            for idx in s:
                consistency[idx] += 1

        if not all_rankings:
            if self.tracer:
                self.tracer.log("retrieve.empty", queries=len(queries))
            return []

        fused = self._rrf(all_rankings)
        num_rankings = len(all_rankings)
        max_rrf = num_rankings / (self.RRF_K + 1)
        n_queries = max(len(queries), 1)

        results = []
        for idx, score in fused:
            if 0 <= idx < len(self.texts):
                results.append({
                    "content": self.texts[idx],
                    "metadata": self.metas[idx],
                    "score": round(score / max_rrf, 4) if max_rrf > 0 else 0,
                    "consistency": round(consistency[idx] / n_queries, 3),
                    "rank": len(results) + 1,
                })

        results = self._dedup(results)[: max(top_k, 20)]
        for i, r in enumerate(results):
            r["rank"] = i + 1

        if self.tracer:
            self.tracer.log(
                "retrieve.hybrid",
                queries=len(queries),
                query_type=primary_query_type,
                bm25_weight=bm25_repeats,
                fetch_k=fetch_k,
                returned=len(results),
                latency_ms=round((time.time() - t0) * 1000, 1),
            )
        return results


# ============================================================
# LAYER 4: SOURCE CONFIDENCE SCORING
# ============================================================
class ConfidenceScorer:
    """
    Combines four signals into a single confidence score in [0, 1]:
      - retrieval_score   : RRF-normalized hybrid retrieval score
      - freshness_score   : exp(-age_days / decay), newer files score higher
      - trust_score       : configurable per-source (e.g. official docs > drafts)
      - consistency       : fraction of query variants that surfaced this chunk

    final = retrieval * (0.4 + 0.2*freshness + 0.2*trust + 0.2*consistency)

    The base 0.4 ensures retrieval score dominates; the 0.6 multiplier comes from
    the three modulators, each weighted equally. Thresholds for fallback are
    chosen on the COMBINED score, not retrieval alone.
    """

    def __init__(
        self,
        trust_map: Optional[Dict[str, float]] = None,
        freshness_decay_days: float = 365.0,
    ):
        self.trust_map = trust_map or {}
        self.decay = freshness_decay_days

    def freshness(self, mtime_iso: Optional[str]) -> float:
        if not mtime_iso:
            return 0.5  # unknown → neutral
        try:
            t = datetime.fromisoformat(mtime_iso.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - t).total_seconds() / 86400.0
            return float(math.exp(-max(age_days, 0) / self.decay))
        except Exception:
            return 0.5

    def trust(self, source_file: str) -> float:
        if not source_file:
            return 0.5
        for pattern, score in self.trust_map.items():
            if pattern in source_file:
                return float(score)
        return 0.5

    def score(self, chunk: Dict) -> float:
        retrieval = float(chunk.get("score", 0.0))
        consistency = float(chunk.get("consistency", 0.0))
        meta = chunk.get("metadata", {})
        freshness = self.freshness(meta.get("ingest_time"))
        trust = self.trust(meta.get("source_file", ""))
        modifier = 0.4 + 0.2 * freshness + 0.2 * trust + 0.2 * consistency
        return round(min(retrieval * modifier, 1.0), 4)

    def annotate(self, chunks: List[Dict]) -> List[Dict]:
        for c in chunks:
            c["confidence"] = self.score(c)
            c["freshness"] = round(self.freshness(c.get("metadata", {}).get("ingest_time")), 3)
            c["trust"] = round(self.trust(c.get("metadata", {}).get("source_file", "")), 3)
        return chunks


# ============================================================
# LAYER 7: HALLUCINATION FALLBACK GUARD
# ============================================================
class HallucinationGuard:
    """
    Decides whether retrieved evidence is strong enough to generate an answer.

    A query falls back to "Insufficient evidence" when ANY of:
      - top retrieval score is below `min_retrieval_score`
      - top combined confidence is below `min_confidence`
      - top chunk's keyword overlap with the query is below `min_keyword_overlap`
        (catches semantic drift where vectors match but content doesn't)
    """

    INSUFFICIENT_MSG = (
        "Insufficient evidence in the indexed documents to answer this question reliably. "
        "I'd rather say I don't know than guess."
    )

    def __init__(
        self,
        min_retrieval_score: float = 0.15,
        min_confidence: float = 0.15,
        min_keyword_overlap: float = 0.05,
    ):
        self.min_retrieval = min_retrieval_score
        self.min_confidence = min_confidence
        self.min_keyword_overlap = min_keyword_overlap

    @staticmethod
    def _keyword_overlap(query: str, content: str) -> float:
        q_tokens = set(re.findall(r"\w+", query.lower())) - _STOPWORDS
        c_tokens = set(re.findall(r"\w+", content.lower()))
        if not q_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / len(q_tokens)

    def check(self, query: str, chunks: List[Dict]) -> Tuple[bool, str]:
        if not chunks:
            return False, "no_chunks"
        top = chunks[0]
        retrieval = float(top.get("score", 0.0))
        confidence = float(top.get("confidence", retrieval))
        top_overlap = self._keyword_overlap(query, top.get("content", ""))

        # Aggregate evidence: average overlap across top-3 chunks
        # Catches cases where a single chunk has a coincidental keyword match
        top3 = chunks[: min(3, len(chunks))]
        avg_overlap = sum(self._keyword_overlap(query, c.get("content", "")) for c in top3) / max(len(top3), 1)

        if retrieval < self.min_retrieval:
            return False, f"low_retrieval({retrieval:.3f})"
        if confidence < self.min_confidence:
            return False, f"low_confidence({confidence:.3f})"
        if top_overlap < self.min_keyword_overlap:
            return False, f"low_top_overlap({top_overlap:.3f})"
        if avg_overlap < self.min_keyword_overlap * 0.7:
            # broader retrieval is mostly off-topic
            return False, f"low_avg_overlap({avg_overlap:.3f})"
        return True, "ok"


_STOPWORDS = set(
    "a an the and or but of in on at to for with by from as is are was were be been being "
    "do does did have has had this that these those it its his her them they we us you your "
    "what which who whom whose how why where when can could would should may might shall will "
    "i me my mine our ours yourself yourselves himself herself itself themselves not no nor "
    "than then so if while because about into through during before after above below up down "
    "out over under again further also very too just only own same other".split()
)


# ============================================================
# LAYER 6: CITATION VERIFIER (per-claim grounding)
# ============================================================
class CitationVerifier:
    """
    Splits the answer into sentences and grounds each one against retrieved chunks.

    For every sentence we compute lexical overlap (TF-style word match) with each
    chunk; the chunk with the highest overlap is the supporting source. Sentences
    with overlap below `min_grounding` are flagged as potentially unsupported.

    Output:
      annotated_answer: each sentence followed by [n] markers
      grounding_score : average per-sentence overlap (0-1)
      flagged         : list of unsupported sentences
    """

    def __init__(self, min_grounding: float = 0.15):
        self.min_grounding = min_grounding

    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall(r"\w+", text.lower())) - _STOPWORDS

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        # Robust to abbreviations, decimals, and quoted phrases
        text = re.sub(r"\s+", " ", text.strip())
        sentences: List[str] = []
        buf = ""
        for token in re.split(r"(?<=[.!?])\s+(?=[A-Z\"\(])", text):
            if token.strip():
                sentences.append(token.strip())
        return sentences if sentences else ([text] if text else [])

    def verify(self, answer: str, chunks: List[Dict]) -> Dict[str, Any]:
        sentences = self._split_sentences(answer)
        if not sentences or not chunks:
            return {
                "annotated": answer,
                "grounding_score": 0.0,
                "flagged": [],
                "per_sentence": [],
            }

        chunk_tokens = [self._tokens(c["content"]) for c in chunks]
        per_sentence: List[Dict] = []
        annotated_parts: List[str] = []
        flagged: List[str] = []

        for sent in sentences:
            stoks = self._tokens(sent)
            if not stoks:
                annotated_parts.append(sent)
                continue
            best_idx, best_overlap = 0, 0.0
            for i, ctoks in enumerate(chunk_tokens):
                if not ctoks:
                    continue
                overlap = len(stoks & ctoks) / max(len(stoks), 1)
                if overlap > best_overlap:
                    best_overlap, best_idx = overlap, i
            per_sentence.append({
                "sentence": sent,
                "support_chunk": best_idx,
                "overlap": round(best_overlap, 3),
                "grounded": best_overlap >= self.min_grounding,
            })
            if best_overlap >= self.min_grounding:
                annotated_parts.append(f"{sent} [{best_idx + 1}]")
            else:
                annotated_parts.append(f"{sent} [?]")
                flagged.append(sent)

        avg = sum(p["overlap"] for p in per_sentence) / max(len(per_sentence), 1)
        return {
            "annotated": " ".join(annotated_parts),
            "grounding_score": round(avg, 3),
            "flagged": flagged,
            "per_sentence": per_sentence,
        }


# ============================================================
# 5. RAG PIPELINE (Multi-Query + Rewrite + Two-Stage Rerank + Guards)
# ============================================================
class RAGPipeline:

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: GroqLLM,
        scorer: Optional[ConfidenceScorer] = None,
        guard: Optional[HallucinationGuard] = None,
        verifier: Optional[CitationVerifier] = None,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        cache: Optional[TTLCache] = None,
        tracer: Optional[TraceLogger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.scorer = scorer or ConfidenceScorer()
        self.guard = guard or HallucinationGuard()
        self.verifier = verifier or CitationVerifier()
        self.cross_encoder = cross_encoder
        self.cache = cache
        self.tracer = tracer
        self.metrics = metrics
        self.history: List[Dict] = []

    def _expand_query(self, question: str) -> List[str]:
        prompt = (
            "Generate 3 different search queries to help answer the question below.\n"
            "Each query should use different keywords and phrasings.\n"
            "If the question asks for a number, score, or statistic, make one query "
            "target table data (e.g. include 'Table', 'results', or metric names).\n"
            "Return ONLY the queries, one per line, numbered 1-3.\n\n"
            f"Question: {question}\n\nQueries:"
        )
        try:
            resp = self.llm.invoke(prompt, max_tokens=200)
            lines = [
                re.sub(r"^\d+[\.\)]\s*", "", l).strip()
                for l in resp.strip().split("\n")
                if l.strip()
            ]
            return [l for l in lines if len(l) > 5][:3]
        except Exception:
            return []

    def _rewrite(self, question: str) -> str:
        if not self.history:
            return question
        # Use ONLY prior user questions, not assistant answers.
        # Including answer text causes facts to leak into the rewrite.
        prior_questions = [t["q"] for t in self.history[-3:]]
        ctx = "\n".join(f"- {q}" for q in prior_questions)
        prompt = (
            "You are a query rewriter. Decide whether to rewrite a new question based on "
            "previous user questions in the same conversation.\n\n"
            "RULES:\n"
            "- If the new question is fully self-contained, return it EXACTLY as-is.\n"
            "- If it is a vague follow-up (e.g. 'tell me more', 'what about it'), rewrite it "
            "into a standalone question using ONLY entity names from the previous questions.\n"
            "- DO NOT include facts, definitions, quotes, or filenames in the rewrite. "
            "Your output is a question for retrieval, not a summary.\n"
            "- DO NOT change the meaning. DO NOT add 'specifically in the context of ...' clauses.\n"
            "- Output MUST be a single question ending with '?'.\n"
            "- If unsure, return the question unchanged.\n\n"
            f"Previous user questions:\n{ctx}\n\n"
            f"New question: {question}\n\n"
            "Rewritten question:"
        )
        try:
            r = self.llm.invoke(prompt, max_tokens=80).strip()
            # Reject rewrites that are too long (sign of fact-leakage) or contain doc names
            if not r or len(r) < 5:
                return question
            if len(r) > len(question) * 2.5:
                return question  # rewriter expanded too much -- discard
            if any(token in r.lower() for token in [".pdf", ".docx", "specifically in the context"]):
                return question  # filename or hedging leaked
            return r
        except Exception:
            return question

    @staticmethod
    def _source_hint_boost(question: str, candidates: List[Dict]) -> List[Dict]:
        """
        Source-aware boost: if the query names a specific paper/document/system,
        boost candidates whose source filename contains those tokens.
        Prevents cross-document contamination where a question about 'RAG paper'
        retrieves chunks from an unrelated 'embedding report'.
        """
        q_tokens = set(re.findall(r"\w+", question.lower())) - _STOPWORDS
        if not q_tokens:
            return candidates
        for c in candidates:
            src = (c.get("metadata", {}).get("source_file") or "").lower()
            src_tokens = set(re.findall(r"\w+", src)) - _STOPWORDS
            overlap = len(q_tokens & src_tokens)
            # multiplicative boost up to 1.5x for clear source matches
            c["score"] = c.get("score", 0) * (1.0 + 0.15 * overlap)
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        return candidates

    def _rerank(self, question: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Two-stage reranking:
          Stage 1: cross-encoder via HF Inference API (cheap, deep, model-driven).
          Stage 2 (fallback): LLM rerank if cross-encoder fails or is disabled.
        Plus: source-hint boost to prevent cross-document contamination.
        """
        # Apply source-hint boost first (cheap, prevents off-topic chunks reaching reranker)
        candidates = self._source_hint_boost(question, candidates)

        if len(candidates) <= top_k:
            return candidates

        # Try cross-encoder first
        if self.cross_encoder is not None:
            passages = [c["content"][:1500] for c in candidates]
            scores = self.cross_encoder.score_pairs(question, passages)
            if scores and len(scores) == len(candidates):
                ordered = sorted(
                    zip(candidates, scores), key=lambda x: x[1], reverse=True
                )
                reranked = [c for c, _ in ordered[:top_k]]
                for i, (c, s) in enumerate(ordered[:top_k]):
                    c["rank"] = i + 1
                    c["ce_score"] = round(float(s), 4)
                if self.tracer:
                    self.tracer.log("rerank.applied", method="cross_encoder", n=len(candidates))
                return reranked

        # Fallback: LLM rerank
        numbered = "\n\n".join(
            f"[{i}] {c['content'][:800]}" for i, c in enumerate(candidates)
        )
        prompt = (
            "Given a question and numbered document chunks, return ONLY the numbers "
            "of the chunks most relevant to answering the question, in order of relevance.\n"
            f"Return exactly {top_k} numbers, comma-separated. No explanation.\n\n"
            f"Question: {question}\n\nChunks:\n{numbered}\n\nRelevant chunk numbers:"
        )
        try:
            resp = self.llm.invoke(prompt, max_tokens=50)
            indices = [int(x.strip().strip("[]")) for x in resp.split(",") if x.strip().strip("[]").isdigit()]
            reranked = []
            for idx in indices[:top_k]:
                if 0 <= idx < len(candidates):
                    reranked.append(candidates[idx])
            if reranked:
                for i, r in enumerate(reranked):
                    r["rank"] = i + 1
                if self.tracer:
                    self.tracer.log("rerank.applied", method="llm", n=len(candidates))
                return reranked
        except Exception:
            pass
        if self.tracer:
            self.tracer.log("rerank.applied", method="passthrough", n=len(candidates))
        return candidates[:top_k]

    @staticmethod
    def _sanitize(text: str, max_len: int = 300) -> str:
        import re as _re
        text = _re.sub(r"<\|[^|]*\|>", "", text)
        text = _re.sub(r"\n{3,}", "\n\n", text).strip()
        return text[:max_len] if len(text) > max_len else text

    def _history_prompt(self) -> str:
        if not self.history:
            return ""
        return (
            "Previous conversation:\n"
            + "\n".join(
                f"User: {t['q']}\nAssistant: {self._sanitize(t['a'])}"
                for t in self.history[-3:]
            )
            + "\n\n"
        )

    @staticmethod
    def _build_constrained_prompt(question: str, history: str, context: str) -> str:
        """Layer 5: Constraint generation -- context-only, no external knowledge, no cross-source synthesis."""
        return (
            "You are a precise research assistant operating in STRICT GROUNDED mode.\n\n"
            "INVIOLABLE RULES:\n"
            "1. Use ONLY information present in the context below. Do NOT use general knowledge.\n"
            "2. ANTI-CONTAMINATION: NEVER combine notation, formulas, model names, or numbers from one SOURCE_FILE with content from another to construct a hybrid statement. If a fact appears in only one chunk, do not pair it with vocabulary from a different chunk.\n"
            "3. Use whichever chunks actually contain the answer, regardless of source -- but each individual claim must come verbatim from a single chunk.\n"
            "4. If the question explicitly names a paper/document (e.g. 'the RAG paper'), prefer chunks whose SOURCE_FILE matches that name when both options exist.\n"
            "5. If no chunk in the context contains the answer, say: \"The provided documents do not contain enough information to answer this question.\"\n"
            "6. Quote key phrases verbatim using double quotes.\n"
            "7. Never speculate, never assume, never fabricate names, numbers, dates, or section headers. Do not invent text that does not appear verbatim in any chunk.\n"
            "8. If multiple chunks report the SAME metric with DIFFERENT values, prefer values from main results / test set tables over ablations, dev sets, or supplementary material.\n"
            "9. Be concise -- 1-4 sentences. State each fact once. No filler. No markdown headers.\n"
            "10. Do NOT generate follow-up questions, suggestions, or instructions.\n\n"
            f"{history}"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "GROUNDED ANSWER:"
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        summarize: bool = False,
        rewrite: bool = True,
        multi_query: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.time()
        if self.tracer:
            self.tracer.reset()
            self.tracer.log("query.start", question=question, top_k=top_k)

        # Layer 9: query-level cache
        cache_key = _cache_key(
            "query", question.lower().strip(), top_k, rewrite, multi_query
        ) if self.cache else None
        if cache_key and not self.history:  # only cache when no convo context
            hit = self.cache.get(cache_key)
            if hit is not None:
                if self.metrics:
                    self.metrics.record_cache(hit=True)
                if self.tracer:
                    self.tracer.log("query.cache_hit")
                hit = dict(hit)
                hit["from_cache"] = True
                return hit
            elif self.metrics:
                self.metrics.record_cache(hit=False)

        original = question
        rewritten = self._rewrite(question) if rewrite else question
        was_rewritten = rewritten != question
        if self.tracer:
            self.tracer.log("query.rewrite", was_rewritten=was_rewritten, rewritten=rewritten)

        queries = [rewritten]
        sub_queries: List[str] = []
        if multi_query:
            sub_queries = self._expand_query(rewritten)
            queries.extend(sub_queries)
        if self.tracer:
            self.tracer.log("query.expand", n_subqueries=len(sub_queries))

        # Stage 1: hybrid retrieval (ANN + BM25 + RRF)
        raw_results = self.retriever.retrieve(queries, top_k=top_k * 4)

        # Stage 2: cross-encoder or LLM reranker
        results = self._rerank(rewritten, raw_results, top_k) if raw_results else []

        # Layer 4: source confidence scoring (freshness * trust * consistency)
        results = self.scorer.annotate(results)
        results = sorted(results, key=lambda r: r["confidence"], reverse=True)

        # Layer 7: hallucination guard (BEFORE generation, not after)
        ok, reason = self.guard.check(rewritten, results)
        if not ok:
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            if self.tracer:
                self.tracer.log("query.fallback", reason=reason, latency_ms=elapsed_ms)
            if self.metrics:
                self.metrics.record_query(elapsed_ms, 0.0, grounded=False, fallback=True)
            return dict(
                answer=HallucinationGuard.INSUFFICIENT_MSG,
                raw_answer=HallucinationGuard.INSUFFICIENT_MSG,
                sources=self._build_sources(results[:top_k]),
                summary=None,
                confidence=0.0,
                grounding_score=0.0,
                fallback=True,
                fallback_reason=reason,
                rewritten_query=rewritten if was_rewritten else None,
                sub_queries=sub_queries or None,
                trace=self.tracer.events if self.tracer else [],
                latency_ms=elapsed_ms,
            )

        # Layer 5: constraint generation (strict prompt)
        # Each chunk header repeats the source filename so the LLM cannot lose track
        # when chunks from multiple documents appear together (cross-source contamination guard)
        context_parts = []
        for i, r in enumerate(results):
            meta = r.get("metadata", {})
            src = meta.get("source_file", "?")
            pg = meta.get("page", "?")
            context_parts.append(
                f"==================== CHUNK {i+1} ====================\n"
                f"SOURCE_FILE: {src}\n"
                f"PAGE: {pg}\n"
                f"---\n"
                f"{r['content']}"
            )
        context = "\n\n".join(context_parts)
        prompt = self._build_constrained_prompt(original, self._history_prompt(), context)
        answer = self.llm.invoke(prompt, max_tokens=512)

        # Layer 6: per-claim citation verification
        verification = self.verifier.verify(answer, results)
        annotated = verification["annotated"]
        grounding_score = verification["grounding_score"]
        flagged = verification["flagged"]
        if self.tracer:
            self.tracer.log(
                "verify",
                grounding_score=grounding_score,
                flagged_count=len(flagged),
            )

        # Build citation block with doc + page + ingest timestamp
        sources = self._build_sources(results)
        cites = "\n".join(
            f"[{i+1}] {s['source']} · page {s['page']} · {s.get('ingest_time', 'unknown')}"
            for i, s in enumerate(sources)
        )

        if flagged and grounding_score < 0.2:
            # Heavy hallucination: replace with insufficient-evidence response
            full_answer = HallucinationGuard.INSUFFICIENT_MSG + "\n\n*(Generated answer flagged as poorly grounded.)*"
            confidence = 0.0
            if self.tracer:
                self.tracer.log("query.post_hoc_fallback", flagged=len(flagged), grounding=grounding_score)
        else:
            full_answer = f"{annotated}\n\n**Citations:**\n{cites}"
            confidence = max((r.get("confidence", 0.0) for r in results), default=0.0)

        summary = None
        if summarize:
            summary = self.llm.invoke(
                f"Summarize the following answer in 2 short sentences:\n{answer}",
                max_tokens=120,
            )

        self.history.append({"q": original, "a": answer})

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        if self.metrics:
            self.metrics.record_query(elapsed_ms, confidence, grounded=(grounding_score >= 0.2), fallback=False)
        if self.tracer:
            self.tracer.log("query.done", latency_ms=elapsed_ms, confidence=confidence)

        result = dict(
            answer=full_answer,
            raw_answer=answer,
            sources=sources,
            summary=summary,
            confidence=confidence,
            grounding_score=grounding_score,
            flagged_sentences=flagged,
            per_sentence_grounding=verification["per_sentence"],
            fallback=False,
            rewritten_query=rewritten if was_rewritten else None,
            sub_queries=sub_queries or None,
            trace=list(self.tracer.events) if self.tracer else [],
            latency_ms=elapsed_ms,
        )

        # cache stable (history-free) results
        if cache_key and not self.history[:-1]:
            self.cache.put(cache_key, result)
        return result

    @staticmethod
    def _build_sources(results: List[Dict]) -> List[Dict]:
        sources = []
        for r in results:
            meta = r.get("metadata", {})
            sources.append({
                "source": meta.get("source_file", meta.get("source", "?")),
                "page": meta.get("page", "N/A"),
                "score": r.get("score", 0.0),
                "confidence": r.get("confidence", r.get("score", 0.0)),
                "ce_score": r.get("ce_score"),
                "freshness": r.get("freshness"),
                "trust": r.get("trust"),
                "consistency": r.get("consistency"),
                "ingest_time": meta.get("ingest_time"),
                "doc_hash": meta.get("doc_hash"),
                "version": meta.get("version"),
                "preview": (r.get("content", "")[:200] + "..."),
            })
        return sources


# ============================================================
# 6. DOCUMENT LOADING & PROCESSING
# ============================================================
class SimpleDoc:
    """Minimal document container replacing LangChain Document."""
    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata


def _normalize_text(text: str) -> str:
    """Standardize whitespace, fix common PDF artifacts."""
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = text.replace("\ufeff", "")  # BOM
    text = re.sub(r"-\n([a-z])", r"\1", text)  # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _file_hash(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fp:
            while True:
                buf = fp.read(chunk)
                if not buf:
                    break
                h.update(buf)
    except Exception:
        return ""
    return h.hexdigest()[:16]


def _file_version(filename: str) -> str:
    """Extract a version label from filename if present (v1, v2.0, _2024_, etc.)."""
    m = re.search(r"[_\s]?[vV](\d+(?:\.\d+)*)", filename)
    if m:
        return f"v{m.group(1)}"
    m = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return "v0"


def load_pdfs(pdf_directory: str, tracer: Optional[TraceLogger] = None):
    """
    Layer 1: Ingest + normalize.
    Attaches per-page metadata: source_file, page, file_type, doc_hash,
    version, mtime, ingest_time, file_size_bytes.
    """
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        return [], 0
    all_docs: List[SimpleDoc] = []
    ingest_time = datetime.now(timezone.utc).isoformat()
    for f in pdf_files:
        try:
            stat = f.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            doc_hash = _file_hash(f)
            version = _file_version(f.name)
            size = stat.st_size
            doc = fitz.open(str(f))
            page_count = len(doc)
            for page_num in range(page_count):
                text = doc[page_num].get_text()
                if text.strip():
                    text = _normalize_text(text)
                    all_docs.append(SimpleDoc(
                        page_content=text,
                        metadata={
                            "source_file": f.name,
                            "page": page_num + 1,
                            "file_type": "pdf",
                            "doc_hash": doc_hash,
                            "version": version,
                            "mtime": mtime,
                            "ingest_time": ingest_time,
                            "file_size_bytes": size,
                            "total_pages": page_count,
                            "source_path": str(f.relative_to(pdf_dir)) if f.is_relative_to(pdf_dir) else f.name,
                        },
                    ))
            doc.close()
            if tracer:
                tracer.log("ingest.pdf", file=f.name, pages=page_count, hash=doc_hash, version=version)
        except Exception as e:
            st.warning(f"Error loading {f.name}: {e}")
            if tracer:
                tracer.log("ingest.error", file=f.name, error=str(e)[:200])
    return all_docs, len(pdf_files)


def split_documents(documents: List[SimpleDoc], chunk_size=1000, chunk_overlap=200) -> List[SimpleDoc]:
    separators = ["\n\n", "\n", ". ", " "]
    chunks: List[SimpleDoc] = []

    # Adaptive chunking: figure out page count per source file
    pages_per_file: Dict[str, int] = {}
    for doc in documents:
        src = doc.metadata.get("source_file", "")
        pages_per_file[src] = pages_per_file.get(src, 0) + 1

    for doc in documents:
        src = doc.metadata.get("source_file", "")
        num_pages = pages_per_file.get(src, 1)
        # Short docs (<=5 pages): use smaller chunks for fine-grained retrieval
        # Long docs (>5 pages): use the configured chunk size
        effective_size = min(chunk_size, 500) if num_pages <= 5 else chunk_size
        effective_overlap = min(chunk_overlap, 100) if num_pages <= 5 else chunk_overlap

        text = doc.page_content
        parts = _recursive_split(text, separators, effective_size, effective_overlap)
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            chunks.append(SimpleDoc(
                page_content=part.strip(),
                metadata={**doc.metadata, "chunk_index": i},
            ))
    return chunks


def _recursive_split(text: str, separators: List[str], chunk_size: int, chunk_overlap: int = 200) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    sep = separators[0] if separators else " "
    remaining_seps = separators[1:] if len(separators) > 1 else []
    parts = text.split(sep)
    result: List[str] = []
    current = ""
    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                result.append(current)
            if len(part) > chunk_size and remaining_seps:
                result.extend(_recursive_split(part, remaining_seps, chunk_size, chunk_overlap))
                current = ""
            else:
                current = part
    if current:
        result.append(current)
    # Add overlap between chunks
    if chunk_overlap > 0 and len(result) > 1:
        overlapped = [result[0]]
        for i in range(1, len(result)):
            prev_tail = result[i - 1][-chunk_overlap:]
            overlapped.append(prev_tail + result[i])
        result = overlapped
    return result


def deduplicate_chunks(chunks):
    seen = set()
    unique = []
    for c in chunks:
        h = hashlib.md5(
            " ".join(c.page_content.split()).strip().lower().encode()
        ).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(c)
    return unique, len(chunks) - len(unique)


def enrich_chunks(chunks) -> List[str]:
    """Prepend source + version metadata to each chunk text for retrieval signal."""
    enriched = []
    for c in chunks:
        src = c.metadata.get("source_file", "unknown")
        pg = c.metadata.get("page", "?")
        ver = c.metadata.get("version", "v0")
        enriched.append(f"[Source: {src} | Page: {pg} | Version: {ver}]\n{c.page_content}")
    return enriched


# ============================================================
# LAYER 8: CONTINUOUS EVALS
# ============================================================
class EvalHarness:
    """
    Lightweight built-in evaluation suite covering:
      - Recall@k: questions with known supporting documents
      - Adversarial: questions whose answer is NOT in the corpus -- must trigger fallback
      - Hallucination rate: average grounding score across answered questions

    The user can also feed their own JSON eval set.
    """

    DEFAULT_ADVERSARIAL = [
        "What is the capital of Mars in 2050?",
        "Who won the FIFA World Cup in 1742?",
        "What is the boiling point of unobtainium under Saturn's atmosphere?",
        "Quote the third paragraph of a document that was never indexed.",
    ]

    def __init__(self, pipeline: "RAGPipeline"):
        self.pipeline = pipeline

    REFUSAL_PATTERNS = [
        "insufficient evidence",
        "do not contain",
        "does not contain",
        "doesn't contain",
        "not contain enough",
        "not enough information",
        "no information",
        "not provided",
        "not specified",
        "not mentioned",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "doesn't mention",
        "does not mention",
        "not found in documents",
        "i don't know",
    ]

    @classmethod
    def _is_honest_refusal(cls, text: str) -> bool:
        if not text:
            return False
        low = text.lower()
        return any(p in low for p in cls.REFUSAL_PATTERNS)

    def run_adversarial(self, questions: Optional[List[str]] = None) -> Dict[str, Any]:
        questions = questions or self.DEFAULT_ADVERSARIAL
        results = []
        correct_fallbacks = 0
        for q in questions:
            r = self.pipeline.query(q, top_k=5, multi_query=False, rewrite=False)
            hard_guard = bool(r.get("fallback"))
            soft_refusal = self._is_honest_refusal(r.get("raw_answer", ""))
            triggered = hard_guard or soft_refusal
            if triggered:
                correct_fallbacks += 1
            results.append({
                "question": q,
                "fallback": triggered,
                "guard_triggered": hard_guard,
                "soft_refusal": soft_refusal and not hard_guard,
                "raw_answer": r.get("raw_answer", "")[:200],
                "confidence": r.get("confidence", 0),
            })
        return {
            "type": "adversarial",
            "total": len(questions),
            "correct_fallbacks": correct_fallbacks,
            "fallback_rate": round(correct_fallbacks / max(len(questions), 1), 3),
            "details": results,
        }

    def run_recall(self, gold: List[Dict]) -> Dict[str, Any]:
        """
        gold = [{"question": str, "expected_keywords": [str, ...], "expected_source": str (optional)}]
        Passes if all expected keywords appear in raw_answer AND expected_source is cited.
        """
        passed = 0
        details = []
        for item in gold:
            r = self.pipeline.query(item["question"], top_k=5)
            answer = r.get("raw_answer", "").lower()
            kw_hits = sum(1 for kw in item.get("expected_keywords", []) if kw.lower() in answer)
            kw_total = len(item.get("expected_keywords", []) or [])
            kw_pass = kw_total == 0 or kw_hits == kw_total
            src_pass = True
            if "expected_source" in item:
                src_pass = any(item["expected_source"].lower() in s.get("source", "").lower() for s in r.get("sources", []))
            ok = kw_pass and src_pass
            if ok:
                passed += 1
            details.append({
                "question": item["question"],
                "passed": ok,
                "keywords_found": f"{kw_hits}/{kw_total}",
                "source_cited": src_pass,
                "grounding_score": r.get("grounding_score", 0),
                "answer_preview": r.get("raw_answer", "")[:200],
            })
        return {
            "type": "recall",
            "total": len(gold),
            "passed": passed,
            "pass_rate": round(passed / max(len(gold), 1), 3),
            "details": details,
        }


# ============================================================
# 7. STREAMLIT UI
# ============================================================
def _save_uploaded_files(uploaded_files, target_dir: str) -> int:
    """Save Streamlit-uploaded PDFs into the data folder."""
    if not uploaded_files:
        return 0
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    for uf in uploaded_files:
        try:
            (Path(target_dir) / uf.name).write_bytes(uf.getbuffer())
            saved += 1
        except Exception as e:
            st.warning(f"Could not save {uf.name}: {e}")
    return saved


def main():
    st.set_page_config(
        page_title="RAG Document Chatbot", page_icon="🤖", layout="wide"
    )

    st.markdown(
        """
    <style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0 .2rem;
    }
    .sub-header { font-size: 1rem; color: #8e8ea0; text-align: center; margin-bottom: 1.5rem; }
    .source-card {
        background: #1e1e2e; border: 1px solid #333; border-radius: 10px;
        padding: 12px 16px; margin: 6px 0; font-size: .85rem;
    }
    .source-file { color: #667eea; font-weight: 600; }
    .confidence-high { color: #4ade80; font-weight: 700; }
    .confidence-mid  { color: #facc15; font-weight: 700; }
    .confidence-low  { color: #f87171; font-weight: 700; }
    .stat-box {
        background: #1e1e2e; border: 1px solid #333; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .stat-number { font-size: 1.8rem; font-weight: 700; color: #667eea; }
    .stat-label  { font-size: .8rem; color: #8e8ea0; }
    .badge {
        border-radius: 8px; padding: 6px 12px;
        font-size: .8rem; margin-bottom: 4px;
    }
    .rewrite-badge { background: #2d2b55; border: 1px solid #667eea; color: #a5b4fc; }
    .subquery-badge { background: #1a2332; border: 1px solid #2d4a5e; color: #7dd3fc; }
    .fallback-banner {
        background: #2a1010; border: 1px solid #f87171; border-radius: 10px;
        padding: 12px 16px; color: #fca5a5; margin: 8px 0;
    }
    .meta-pill {
        display: inline-block; background: #14213d; color: #a3bffa;
        border: 1px solid #2d4a8e; border-radius: 12px;
        padding: 2px 8px; font-size: .7rem; margin: 2px 3px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header">🤖 Production RAG Pipeline</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Hybrid · Cross-Encoder Rerank · Confidence-Scored · Hallucination-Guarded · Citation-Verified</div>',
        unsafe_allow_html=True,
    )

    # initialize singleton infrastructure
    if "tracer" not in st.session_state:
        st.session_state["tracer"] = TraceLogger()
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = MetricsCollector()
    if "llm_cache" not in st.session_state:
        st.session_state["llm_cache"] = TTLCache(max_size=512, ttl_seconds=3600)
    if "emb_cache" not in st.session_state:
        st.session_state["emb_cache"] = TTLCache(max_size=4096, ttl_seconds=7 * 24 * 3600)
    if "query_cache" not in st.session_state:
        st.session_state["query_cache"] = TTLCache(max_size=128, ttl_seconds=600)
    tracer: TraceLogger = st.session_state["tracer"]
    metrics: MetricsCollector = st.session_state["metrics"]

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("📁 Data Source")
        data_folder = st.text_input("PDF Folder Path", value="data/pdf_files")
        uploaded = st.file_uploader(
            "Or drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Files are saved to the folder above",
        )
        if uploaded:
            n = _save_uploaded_files(uploaded, data_folder)
            if n:
                st.success(f"📥 Saved {n} file(s) to {data_folder}")

        st.subheader("🔧 Chunking")
        chunk_size = st.slider("Chunk Size", 200, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, step=50)

        st.subheader("🔍 Retrieval")
        top_k = st.slider("Top K Results", 1, 10, 5)
        use_cross_encoder = st.checkbox(
            "Cross-encoder reranker",
            value=True,
            help="Two-stage rerank via HF Inference API (falls back to LLM if unreachable)",
        )

        st.subheader("🛡️ Hallucination Guard")
        min_retrieval = st.slider("Min retrieval score", 0.0, 1.0, 0.15, 0.05)
        min_grounding = st.slider("Min grounding overlap", 0.0, 1.0, 0.05, 0.05)

        st.subheader("🧠 Generation")
        model_name = st.selectbox(
            "Groq Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        )
        enable_summary = st.checkbox("Enable Summary", value=False)
        enable_rewrite = st.checkbox(
            "Enable Query Rewriting",
            value=True,
            help="Rewrites follow-up questions into standalone queries",
        )
        enable_multi_query = st.checkbox(
            "Enable Multi-Query Expansion",
            value=True,
            help="Generates 3 query variations for broader recall",
        )

        st.divider()

        # ---- Load & Index ----
        if st.button(
            "📂 Load & Index Documents", use_container_width=True, type="primary"
        ):
            t0 = time.time()
            tracer.reset()
            tracer.log("index.start", folder=data_folder)

            with st.spinner("Loading & normalizing PDFs..."):
                all_docs, num_files = load_pdfs(data_folder, tracer=tracer)

            if not all_docs:
                st.error(f"No PDFs found in '{data_folder}'.")
            else:
                st.info(f"Loaded {len(all_docs)} pages from {num_files} PDFs.")

                with st.spinner("Splitting into chunks..."):
                    chunks = split_documents(all_docs, chunk_size, chunk_overlap)
                    st.info(f"{len(chunks)} chunks created.")
                    tracer.log("index.chunks", n=len(chunks), chunk_size=chunk_size)

                with st.spinner("Deduplicating..."):
                    chunks, n_dupes = deduplicate_chunks(chunks)
                    if n_dupes:
                        st.info(f"Removed {n_dupes} duplicates → {len(chunks)} unique.")
                    tracer.log("index.dedup", duplicates_removed=n_dupes, unique=len(chunks))

                with st.spinner("Enriching chunks with metadata context..."):
                    enriched = enrich_chunks(chunks)

                with st.spinner("Generating embeddings (cached)..."):
                    emb_mgr = EmbeddingManager(
                        cache=st.session_state["emb_cache"],
                        tracer=tracer,
                        metrics=metrics,
                    )
                    embeddings = emb_mgr.generate_embeddings(enriched)

                with st.spinner("Building vector store (ChromaDB)..."):
                    vs = VectorStore(tracer=tracer)
                    metas = [dict(c.metadata) for c in chunks]
                    vs.add(enriched, embeddings, metas)

                with st.spinner("Building BM25 keyword index..."):
                    bm25 = BM25Index()
                    bm25.build(enriched)

                with st.spinner("Assembling production pipeline..."):
                    retriever = HybridRetriever(vs, emb_mgr, bm25, enriched, metas, tracer=tracer)
                    llm = GroqLLM(
                        api_key=os.getenv("GROQ_API_KEY"),
                        model=model_name,
                        temperature=0.1,
                        max_tokens=1024,
                        cache=st.session_state["llm_cache"],
                        tracer=tracer,
                        metrics=metrics,
                    )
                    cross_enc = None
                    if use_cross_encoder and os.getenv("HF_API_TOKEN"):
                        cross_enc = CrossEncoderReranker(
                            api_key=os.getenv("HF_API_TOKEN"),
                            cache=st.session_state["llm_cache"],
                            tracer=tracer,
                        )
                    scorer = ConfidenceScorer(trust_map={
                        "policy": 1.0, "official": 1.0,
                        "draft": 0.6, "preview": 0.6,
                    })
                    guard = HallucinationGuard(
                        min_retrieval_score=min_retrieval,
                        min_keyword_overlap=min_grounding,
                    )
                    verifier = CitationVerifier(min_grounding=min_grounding)
                    pipeline = RAGPipeline(
                        retriever=retriever,
                        llm=llm,
                        scorer=scorer,
                        guard=guard,
                        verifier=verifier,
                        cross_encoder=cross_enc,
                        cache=st.session_state["query_cache"],
                        tracer=tracer,
                        metrics=metrics,
                    )

                elapsed = round(time.time() - t0, 1)
                tracer.log("index.done", elapsed_s=elapsed)
                st.session_state.update(
                    pipeline=pipeline,
                    indexed=True,
                    num_chunks=len(chunks),
                    num_dupes=n_dupes,
                    num_files=num_files,
                    num_pages=len(all_docs),
                    model_name=model_name,
                    source_files=list(
                        set(c.metadata.get("source_file", "") for c in chunks)
                    ),
                )
                st.success(
                    f"✅ Indexed {len(chunks)} unique chunks from {num_files} docs "
                    f"({n_dupes} duplicates removed) in {elapsed}s"
                )

        st.divider()

        # ---- Status panel ----
        if st.session_state.get("indexed"):
            st.success("✅ Ready to chat")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<div class="stat-box"><div class="stat-number">'
                    f'{st.session_state.get("num_chunks", 0)}</div>'
                    f'<div class="stat-label">Unique Chunks</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="stat-box"><div class="stat-number">'
                    f'{st.session_state.get("num_dupes", 0)}</div>'
                    f'<div class="stat-label">Dupes Removed</div></div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                f"📄 {st.session_state.get('num_files', 0)} files · "
                f"{st.session_state.get('num_pages', 0)} pages"
            )
            st.caption(f"🧠 {st.session_state.get('model_name', '')}")
            if st.session_state.get("source_files"):
                with st.expander("📑 Indexed Documents"):
                    for f in st.session_state["source_files"]:
                        st.markdown(f"- `{f}`")
        else:
            st.info("📌 Load documents to start chatting")

        st.divider()

        if st.session_state.get("messages"):
            st.download_button(
                "📥 Export Chat",
                json.dumps(
                    st.session_state["messages"], indent=2, default=str
                ),
                "chat_history.json",
                "application/json",
                use_container_width=True,
            )

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            if "pipeline" in st.session_state:
                st.session_state["pipeline"].history = []
            st.rerun()

    # ---- Chat messages ----
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    def _render_result(msg_or_result):
        if msg_or_result.get("fallback"):
            st.markdown(
                f'<div class="fallback-banner">🛡️ Hallucination guard triggered '
                f"({msg_or_result.get('fallback_reason', 'low_evidence')}). "
                "Returning honest fallback rather than guessing.</div>",
                unsafe_allow_html=True,
            )
        if msg_or_result.get("rewritten_query"):
            st.markdown(
                f'<div class="badge rewrite-badge">🔄 Rewritten: '
                f'<strong>{msg_or_result["rewritten_query"]}</strong></div>',
                unsafe_allow_html=True,
            )
        if msg_or_result.get("sub_queries"):
            with st.expander("🔀 Sub-queries generated"):
                for sq in msg_or_result["sub_queries"]:
                    st.markdown(
                        f'<div class="badge subquery-badge">{sq}</div>',
                        unsafe_allow_html=True,
                    )

    def _render_sources(sources):
        if not sources:
            return
        with st.expander(f"📄 Sources ({len(sources)})"):
            for i, src in enumerate(sources):
                sc = src.get("confidence", src.get("score", 0))
                cls = (
                    "confidence-high"
                    if sc >= 0.5
                    else ("confidence-mid" if sc >= 0.25 else "confidence-low")
                )
                pills = []
                if src.get("ce_score") is not None:
                    pills.append(f'<span class="meta-pill">CE {src["ce_score"]}</span>')
                if src.get("freshness") is not None:
                    pills.append(f'<span class="meta-pill">fresh {src["freshness"]}</span>')
                if src.get("trust") is not None:
                    pills.append(f'<span class="meta-pill">trust {src["trust"]}</span>')
                if src.get("consistency") is not None:
                    pills.append(f'<span class="meta-pill">cons {src["consistency"]}</span>')
                if src.get("version"):
                    pills.append(f'<span class="meta-pill">{src["version"]}</span>')
                pills_html = "".join(pills)
                ingest = src.get("ingest_time", "")[:19] if src.get("ingest_time") else ""
                st.markdown(
                    f'<div class="source-card">'
                    f'<span class="source-file">[{i+1}] 📑 {src["source"]}</span>'
                    f' · Page {src["page"]}'
                    f' <span class="{cls}"> · Conf: {sc:.3f}</span>'
                    f'<br>{pills_html}'
                    f'<br><small style="color:#888">ingest: {ingest}</small>'
                    f'<br><small style="color:#666">{src.get("preview", "")[:200]}</small>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    def _render_confidence(result):
        confidence = result.get("confidence", 0) if isinstance(result, dict) else result
        grounding = result.get("grounding_score") if isinstance(result, dict) else None
        if confidence and confidence > 0:
            icon = "🟢" if confidence >= 0.5 else ("🟡" if confidence >= 0.25 else "🔴")
            line = f"{icon} Confidence: {confidence:.3f}"
            if grounding is not None:
                gicon = "🟢" if grounding >= 0.4 else ("🟡" if grounding >= 0.2 else "🔴")
                line += f"  ·  {gicon} Grounding: {grounding:.3f}"
            if isinstance(result, dict) and result.get("latency_ms"):
                line += f"  ·  ⚡ {result['latency_ms']} ms"
            st.caption(line)

    tab_chat, tab_eval, tab_trace, tab_metrics = st.tabs(
        ["💬 Chat", "🧪 Evals", "🔬 Trace", "📊 Metrics"]
    )

    # ============================================================
    # CHAT TAB
    # ============================================================
    with tab_chat:
        for msg in st.session_state["messages"]:
            avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                _render_result(msg)
                st.markdown(msg["content"])
                _render_sources(msg.get("sources"))
                if msg.get("summary"):
                    with st.expander("📝 Summary"):
                        st.markdown(msg["summary"])
                _render_confidence(msg)

        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.get("indexed"):
                st.warning("⚠️ Load documents first.")
            else:
                st.session_state["messages"].append(
                    {"role": "user", "content": prompt}
                )
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="🤖"):
                    pipeline = st.session_state["pipeline"]
                    with st.spinner("🔍 Retrieving · Reranking · Verifying..."):
                        result = pipeline.query(
                            prompt,
                            top_k=top_k,
                            summarize=enable_summary,
                            rewrite=enable_rewrite,
                            multi_query=enable_multi_query,
                        )

                    _render_result(result)
                    st.markdown(result["answer"])
                    _render_sources(result["sources"])
                    if result.get("flagged_sentences"):
                        with st.expander(f"⚠️ {len(result['flagged_sentences'])} ungrounded sentence(s)"):
                            for s in result["flagged_sentences"]:
                                st.markdown(f"- {s}")
                    if result.get("summary"):
                        with st.expander("📝 Summary"):
                            st.markdown(result["summary"])
                    _render_confidence(result)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "summary": result.get("summary"),
                    "confidence": result.get("confidence"),
                    "grounding_score": result.get("grounding_score"),
                    "latency_ms": result.get("latency_ms"),
                    "fallback": result.get("fallback"),
                    "fallback_reason": result.get("fallback_reason"),
                    "rewritten_query": result.get("rewritten_query"),
                    "sub_queries": result.get("sub_queries"),
                    "flagged_sentences": result.get("flagged_sentences"),
                })

    # ============================================================
    # EVAL TAB
    # ============================================================
    with tab_eval:
        st.subheader("🧪 Continuous Evaluations")
        st.caption(
            "Adversarial: questions whose answer is NOT in your corpus -- should fall back. "
            "Recall: questions with known expected keywords/sources."
        )

        if not st.session_state.get("indexed"):
            st.info("Load documents first to run evals.")
        else:
            pipeline = st.session_state["pipeline"]
            harness = EvalHarness(pipeline)

            colA, colB = st.columns(2)
            with colA:
                if st.button("Run Adversarial Suite", use_container_width=True):
                    with st.spinner("Running adversarial probes..."):
                        adv = harness.run_adversarial()
                    st.session_state["last_adv"] = adv

            with colB:
                gold_text = st.text_area(
                    "Recall set (JSON)",
                    value=json.dumps([
                        {
                            "question": "What does POSH stand for?",
                            "expected_keywords": ["Prevention", "Sexual Harassment"],
                            "expected_source": "Anti-Harassment",
                        },
                    ], indent=2),
                    height=150,
                )
                if st.button("Run Recall Eval", use_container_width=True):
                    try:
                        gold = json.loads(gold_text)
                        with st.spinner("Running recall eval..."):
                            rec = harness.run_recall(gold)
                        st.session_state["last_rec"] = rec
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")

            adv = st.session_state.get("last_adv")
            if adv:
                st.markdown(
                    f"### Adversarial · {adv['correct_fallbacks']}/{adv['total']} "
                    f"correct refusals ({adv['fallback_rate'] * 100:.1f}%)"
                )
                st.caption(
                    "✅ = system refused to hallucinate (either via hard guard or honest refusal). "
                    "🛡️ = guard short-circuit (saves an LLM call). "
                    "📜 = LLM produced an honest refusal."
                )
                for d in adv["details"]:
                    if d["guard_triggered"]:
                        icon = "✅ 🛡️"
                    elif d.get("soft_refusal"):
                        icon = "✅ 📜"
                    else:
                        icon = "❌"
                    st.markdown(f"{icon} **{d['question']}**")
                    st.caption(d["raw_answer"])

            rec = st.session_state.get("last_rec")
            if rec:
                st.markdown(
                    f"### Recall · {rec['passed']}/{rec['total']} passed "
                    f"({rec['pass_rate'] * 100:.1f}%)"
                )
                for d in rec["details"]:
                    icon = "✅" if d["passed"] else "❌"
                    st.markdown(
                        f"{icon} **{d['question']}** — kw {d['keywords_found']}, "
                        f"source {'✓' if d['source_cited'] else '✗'}, "
                        f"grounding {d['grounding_score']}"
                    )
                    st.caption(d["answer_preview"])

    # ============================================================
    # TRACE TAB
    # ============================================================
    with tab_trace:
        st.subheader("🔬 Last query trace")
        events = tracer.events
        if not events:
            st.info("No trace yet. Run a query.")
        else:
            durs = tracer.stage_durations()
            cols = st.columns(min(len(durs), 6) or 1)
            for i, (stage, d) in enumerate(sorted(durs.items(), key=lambda x: -x[1])[:6]):
                with cols[i % len(cols)]:
                    st.metric(stage, f"{d * 1000:.0f} ms")
            with st.expander("Full event log", expanded=True):
                st.json(events)
            st.download_button(
                "📥 Export trace (JSON)",
                tracer.to_json(),
                "trace.json",
                "application/json",
                use_container_width=True,
            )

    # ============================================================
    # METRICS TAB
    # ============================================================
    with tab_metrics:
        st.subheader("📊 Pipeline metrics (session)")
        s = metrics.summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Queries", s["queries_total"])
        c2.metric("Grounded %", f"{s['grounded_pct']}%")
        c3.metric("Fallback %", f"{s['fallback_pct']}%")
        c4.metric("Avg confidence", s["avg_confidence"])
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("p50 latency", f"{s['p50_latency_ms']} ms")
        c6.metric("p95 latency", f"{s['p95_latency_ms']} ms")
        c7.metric("Cache hit rate", f"{s['cache_hit_rate'] * 100:.1f}%")
        c8.metric(
            "Cache size",
            f"L:{st.session_state['llm_cache'].size} · "
            f"E:{st.session_state['emb_cache'].size} · "
            f"Q:{st.session_state['query_cache'].size}",
        )
        st.caption(
            "L = LLM response cache · E = embedding cache · Q = query result cache. "
            "Caches are session-scoped and TTL-bounded."
        )
        if st.button("Clear all caches"):
            st.session_state["llm_cache"].clear()
            st.session_state["emb_cache"].clear()
            st.session_state["query_cache"].clear()
            st.success("Caches cleared.")


if __name__ == "__main__":
    main()
