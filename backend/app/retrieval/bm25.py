"""
BM25 (Okapi) over one patient's chunks, with document frequencies from the WHOLE chunk collection.

Per-patient collections are tiny (tens of chunks), so IDF computed only inside them would be noise; the global
DF table is cached and refreshed when the collection changes. Term weights let expansion terms count less
than the words the nurse actually typed.
"""
import math
import threading
import time
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.models import ClinicalDocument, DocumentChunk
from backend.app.retrieval.normalize import tokenize

K1, B = 1.5, 0.75


class GlobalStats:
    """Document frequencies over all READY chunks. Recomputed when the chunk count changes (checked at most every 30 s)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.df: Counter = Counter()
        self.n_docs = 0
        self.avg_len = 1.0
        self._fingerprint = None
        self._checked = 0.0

    def refresh(self, db: Session, force: bool = False) -> None:
        if not force and time.monotonic() - self._checked < 30:
            return
        fp = db.query(func.count(DocumentChunk.id), func.max(DocumentChunk.created_at)).one()
        self._checked = time.monotonic()
        if not force and fp == self._fingerprint:
            return
        texts = [t for (t,) in db.query(DocumentChunk.text).join(ClinicalDocument, ClinicalDocument.id == DocumentChunk.document_id)
                 .filter(ClinicalDocument.status == "READY")]
        df: Counter = Counter()
        total = 0
        for t in texts:
            toks = tokenize(t)
            total += len(toks)
            df.update(set(toks))
        with self._lock:
            self.df, self.n_docs = df, max(len(texts), 1)
            self.avg_len = total / max(len(texts), 1) or 1.0
            self._fingerprint = fp

    def idf(self, term: str) -> float:
        n, d = self.n_docs, self.df.get(term, 0)
        return math.log(1 + (n - d + 0.5) / (d + 0.5))


GLOBAL = GlobalStats()


def bm25_scores(chunk_texts: Sequence[str], query_weights: Dict[str, float], stats: GlobalStats = GLOBAL) -> List[float]:
    docs = [Counter(tokenize(t)) for t in chunk_texts]
    out = []
    for tf in docs:
        length = sum(tf.values())
        s = 0.0
        for term, w in query_weights.items():
            f = tf.get(term, 0)
            if f:
                s += w * stats.idf(term) * f * (K1 + 1) / (f + K1 * (1 - B + B * length / stats.avg_len))
        out.append(s)
    return out


def rank(scores: Sequence[float]) -> List[Tuple[int, float]]:
    """Indices with a positive score, best first."""
    return sorted(((i, s) for i, s in enumerate(scores) if s > 0), key=lambda x: -x[1])
