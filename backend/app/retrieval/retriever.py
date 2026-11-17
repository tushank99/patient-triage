"""
Patient-history retriever (Phase 4).

  query ─► normalise/expand (medical lexicon, multi-query variants)
        ─► dense: every variant embedded, cosine vs the patient's chunks, best variant per chunk
        ─► sparse: BM25 over the patient's chunks (global IDF; expansion terms weighted lower)
        ─► Reciprocal Rank Fusion
        ─► optional cross-encoder rerank of the top candidates
        ─► low-confidence gate (threshold calibrated on the labelled benchmark)
        ─► parent-child: return the chunk's whole SECTION, compressed to a character budget around the hit

HARD PATIENT FILTER: the only entry point takes a patient id and every stage works on that patient's chunk set
(loaded with WHERE patient_id = ...). There is no code path that scores another patient's text. Caches are keyed
by patient id. tests/test_retrieval.py attacks this adversarially.
"""
import json
import math
import os
import re
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.ingestion.embeddings import from_bytes, get_embedder
from backend.app.models import ClinicalDocument, DocumentChunk, DocumentVersion
from backend.app.retrieval import bm25
from backend.app.retrieval.cache import PATIENT_CHUNKS, QUERY_EMBEDDINGS
from backend.app.retrieval.normalize import NormalizedQuery, normalize_query, tokenize

THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "thresholds.json")
DEFAULT_THRESHOLDS = {"rerank": 0.10, "dense": 0.15, "bm25": 0.30}  # used only when no calibrated value exists


@dataclass(frozen=True)
class RetrievalConfig:
    name: str = "hybrid+expansion"
    dense: bool = True
    sparse: bool = True
    expand: bool = True
    rerank: bool = True
    parent_child: bool = True
    candidates: int = 30
    rerank_top: int = 20
    rrf_k: int = 60
    top_k: int = 5
    context_chars: int = 900

    @classmethod
    def from_settings(cls) -> "RetrievalConfig":
        mode = settings.retrieval_mode
        return cls(
            name=f"{mode}{'+expansion' if settings.retrieval_query_expansion else ''}",
            dense=mode in ("hybrid", "vector"), sparse=mode in ("hybrid", "bm25"),
            expand=settings.retrieval_query_expansion, rerank=bool(settings.reranker_model),
            parent_child=settings.retrieval_parent_child, candidates=settings.retrieval_candidates,
            context_chars=settings.retrieval_context_chars,
        )


@dataclass
class Passage:
    chunk_id: str
    document_id: str
    version_id: str
    section: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    chunk_text: str
    context_text: str
    context_kind: str  # chunk | section | section-compressed
    relevance: float  # the score the low-confidence gate looks at (see _relevance)
    ranks: Dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    passages: List[Passage]
    ranked_chunk_ids: List[str]  # full post-rerank order (benchmark metrics use this)
    low_confidence: bool
    top_relevance: Optional[float]
    threshold: float
    threshold_calibrated: bool
    signature: str
    query: NormalizedQuery
    timings_ms: Dict[str, float]
    reason: str = ""


@dataclass
class _PatientChunks:
    ids: List[str]
    document_ids: List[str]
    version_ids: List[str]
    sections: List[Optional[str]]
    section_idx: List[int]
    page_start: List[Optional[int]]
    page_end: List[Optional[int]]
    char_start: List[int]
    char_end: List[int]
    texts: List[str]
    matrix: Optional[np.ndarray]


def _load_patient_chunks(db: Session, patient_id: str, embedding_model: str) -> _PatientChunks:
    base = (db.query(DocumentChunk).join(ClinicalDocument, ClinicalDocument.id == DocumentChunk.document_id)
            .filter(DocumentChunk.patient_id == patient_id, ClinicalDocument.status == "READY",
                    DocumentChunk.embedding_model == embedding_model))
    fp = base.with_entities(func.count(DocumentChunk.id), func.max(DocumentChunk.created_at)).one()

    def load() -> _PatientChunks:
        rows = base.order_by(DocumentChunk.document_id, DocumentChunk.chunk_index).all()
        assert all(r.patient_id == patient_id for r in rows)  # belt and braces: the WHERE clause already did this
        return _PatientChunks(
            ids=[r.id for r in rows], document_ids=[r.document_id for r in rows], version_ids=[r.version_id for r in rows],
            sections=[r.section for r in rows], section_idx=[r.section_index for r in rows],
            page_start=[r.page_start for r in rows], page_end=[r.page_end for r in rows],
            char_start=[r.char_start for r in rows], char_end=[r.char_end for r in rows], texts=[r.text for r in rows],
            matrix=np.vstack([from_bytes(r.embedding) for r in rows]) if rows else None,
        )

    return PATIENT_CHUNKS.get_or_set((patient_id, embedding_model, fp[0], str(fp[1])), load)


def _embed(texts: Sequence[str]) -> np.ndarray:
    embedder = get_embedder()
    return np.vstack([QUERY_EMBEDDINGS.get_or_set((embedder.name, t), lambda t=t: embedder.embed([t])[0]) for t in texts])


def _rrf(rankings: List[List[int]], k: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for ranking in rankings:
        for r, idx in enumerate(ranking):
            out[idx] = out.get(idx, 0.0) + 1.0 / (k + r + 1)
    return out


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def signature(cfg: RetrievalConfig, reranker_name: Optional[str]) -> str:
    mode = "hybrid" if cfg.dense and cfg.sparse else ("vector" if cfg.dense else "bm25")
    rr = reranker_name if (cfg.rerank and reranker_name) else "none"
    return f"{mode}|{get_embedder().name}|{rr}|expand={cfg.expand}"


def load_threshold(sig: str, kind: str) -> tuple[float, bool]:
    try:
        with open(THRESHOLDS_PATH, encoding="utf-8") as f:
            entry = json.load(f).get(sig)
        if entry:
            return float(entry["threshold"]), True
    except (OSError, ValueError):
        pass
    return DEFAULT_THRESHOLDS[kind], False


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def _compress(parent: str, child: str, weights: Dict[str, float], budget: int) -> tuple[str, str]:
    """Keep the hit chunk, then fill the budget with the parent's most query-relevant sentences, in document order."""
    if len(parent) <= budget:
        return parent, "section"
    if len(child) >= budget:
        return child[:budget], "chunk"
    sentences = [s.strip() for s in _SENT.split(parent) if s.strip()]
    scored = []
    for i, s in enumerate(sentences):
        if s in child:
            continue
        toks = set(tokenize(s))
        scored.append((sum(w for t, w in weights.items() if t in toks), -i, i, s))
    keep, used = [], len(child)
    for score, _neg, i, s in sorted(scored, reverse=True):
        if score <= 0 or used + len(s) + 1 > budget:
            continue
        keep.append((i, s))
        used += len(s) + 1
    first_child_pos = next((i for i, s in enumerate(sentences) if s in child), len(sentences))
    before = [s for i, s in sorted(keep) if i < first_child_pos]
    after = [s for i, s in sorted(keep) if i >= first_child_pos]
    return " … ".join(p for p in (" ".join(before), child, " ".join(after)) if p), "section-compressed"


def retrieve(db: Session, patient_id: str, query: str, cfg: Optional[RetrievalConfig] = None, reranker=None) -> RetrievalResult:
    if not patient_id or not isinstance(patient_id, str):
        raise ValueError("retrieve() requires a patient_id: cross-patient retrieval is not supported")
    cfg = cfg or RetrievalConfig.from_settings()
    if reranker is None and cfg.rerank:
        from backend.app.retrieval.models import get_reranker

        reranker = get_reranker()
    if reranker is None:
        cfg = replace(cfg, rerank=False)
    t: Dict[str, float] = {}
    t0 = time.perf_counter()
    nq = normalize_query(query, expand=cfg.expand)
    embedder = get_embedder()
    pc = _load_patient_chunks(db, patient_id, embedder.name)
    sig = signature(cfg, getattr(reranker, "name", None))
    kind = "rerank" if cfg.rerank else ("dense" if cfg.dense else "bm25")
    threshold, calibrated = load_threshold(sig, kind)
    t["load"] = (time.perf_counter() - t0) * 1000
    if not pc.ids:
        return RetrievalResult([], [], True, None, threshold, calibrated, sig, nq, t, reason="no indexed chunks for this patient")

    n = len(pc.ids)
    rankings: List[List[int]] = []
    dense = np.full(n, -1.0)
    if cfg.dense:
        t1 = time.perf_counter()
        q = _embed(nq.variants)
        dense = (pc.matrix @ q.T).max(axis=1)
        rankings.append([int(i) for i in np.argsort(-dense)[: cfg.candidates]])
        t["dense"] = (time.perf_counter() - t1) * 1000
    sparse = np.zeros(n)
    if cfg.sparse:
        t1 = time.perf_counter()
        bm25.GLOBAL.refresh(db)
        weights = nq.weighted_tokens() if cfg.expand else {tok: 1.0 for tok in tokenize(nq.original)}
        sparse = np.array(bm25.bm25_scores(pc.texts, weights))
        rankings.append([i for i, _s in bm25.rank(sparse)][: cfg.candidates])
        t["bm25"] = (time.perf_counter() - t1) * 1000

    fused = _rrf(rankings, cfg.rrf_k)
    order = sorted(fused, key=lambda i: -fused[i])[: cfg.candidates]
    ce: Dict[int, float] = {}
    if cfg.rerank and order:
        t1 = time.perf_counter()
        head = order[: cfg.rerank_top]
        scores = reranker.score(nq.variants[1] if len(nq.variants) > 1 and cfg.expand else nq.original, [pc.texts[i] for i in head])
        ce = {i: float(s) for i, s in zip(head, scores)}
        order = sorted(head, key=lambda i: -ce[i]) + order[cfg.rerank_top:]
        t["rerank"] = (time.perf_counter() - t1) * 1000

    def relevance(i: int) -> float:
        if cfg.rerank and i in ce:
            return _sigmoid(ce[i])
        if cfg.dense:
            return float(dense[i])
        return float(sparse[i] / (sparse[i] + 3.0))

    top_rel = relevance(order[0]) if order else None
    low = top_rel is None or top_rel < threshold

    # parent-child expansion (only for what we return)
    t1 = time.perf_counter()
    weights = nq.weighted_tokens()
    passages: List[Passage] = []
    seen_sections = set()
    version_text: Dict[str, str] = {}
    if cfg.parent_child and order:
        vids = {pc.version_ids[i] for i in order[: cfg.top_k * 3]}
        version_text = {v: txt or "" for v, txt in db.query(DocumentVersion.id, DocumentVersion.extracted_text).filter(DocumentVersion.id.in_(vids))}
    for i in order:
        if len(passages) >= cfg.top_k:
            break
        key = (pc.version_ids[i], pc.section_idx[i])
        if cfg.parent_child and key in seen_sections:
            continue  # another chunk of the same section is already returned with its parent context
        seen_sections.add(key)
        context, kind_ = pc.texts[i], "chunk"
        if cfg.parent_child and version_text.get(pc.version_ids[i]):
            members = [j for j in range(n) if (pc.version_ids[j], pc.section_idx[j]) == key]
            start, end = min(pc.char_start[j] for j in members), max(pc.char_end[j] for j in members)
            parent = version_text[pc.version_ids[i]][start:end]
            context, kind_ = _compress(parent, pc.texts[i], weights, cfg.context_chars)
        passages.append(Passage(
            chunk_id=pc.ids[i], document_id=pc.document_ids[i], version_id=pc.version_ids[i], section=pc.sections[i],
            page_start=pc.page_start[i], page_end=pc.page_end[i], chunk_text=pc.texts[i], context_text=context,
            context_kind=kind_, relevance=round(relevance(i), 4),
            ranks={"dense": float(dense[i]) if cfg.dense else None, "bm25": float(sparse[i]) if cfg.sparse else None,
                   "rerank": ce.get(i), "fused": fused.get(i)},
        ))
    t["context"] = (time.perf_counter() - t1) * 1000
    t["total"] = (time.perf_counter() - t0) * 1000
    return RetrievalResult(
        passages=passages, ranked_chunk_ids=[pc.ids[i] for i in order], low_confidence=low, top_relevance=top_rel,
        threshold=threshold, threshold_calibrated=calibrated, signature=sig, query=nq, timings_ms={k: round(v, 2) for k, v in t.items()},
        reason="best match below the calibrated relevance threshold" if low else "",
    )
