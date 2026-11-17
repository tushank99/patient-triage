"""
Vector index behind one interface. The DATABASE (document_chunks) is always the source of truth for chunk
text, metadata and vectors; an index can be rebuilt from it at any time (`reindex --all`).

  sql     (default) cosine search over the patient's own chunks, loaded from document_chunks. Persistent
          wherever the database is (Neon on Render), transactional with deletes, no extra service.
          Patient-scoped retrieval touches tens of chunks, so brute force is exact and fast.
  chroma  persistent Chroma at CHROMA_PATH, holding the same vectors + metadata. Use it locally or with a
          paid persistent disk: on Render's free tier the folder is wiped on every redeploy.

There is deliberately NO query method without a patient_id: cross-patient retrieval is impossible by API.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Protocol, Sequence

import numpy as np
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.ingestion.embeddings import from_bytes
from backend.app.models import ClinicalDocument, DocumentChunk


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    score: float


class VectorIndex(Protocol):
    name: str

    def upsert(self, db: Session, chunks: Sequence[DocumentChunk]) -> None: ...
    def delete_document(self, db: Session, document_id: str) -> None: ...
    def query(self, db: Session, patient_id: str, vector: np.ndarray, k: int, embedding_model: str) -> List[Hit]: ...


class SqlVectorIndex:
    name = "sql"

    def upsert(self, db: Session, chunks: Sequence[DocumentChunk]) -> None:
        pass  # vectors already live on the chunk rows

    def delete_document(self, db: Session, document_id: str) -> None:
        pass  # chunk rows are deleted by the caller in the same transaction

    def query(self, db: Session, patient_id: str, vector: np.ndarray, k: int, embedding_model: str) -> List[Hit]:
        rows = (
            db.query(DocumentChunk.id, DocumentChunk.document_id, DocumentChunk.embedding)
            .join(ClinicalDocument, ClinicalDocument.id == DocumentChunk.document_id)
            .filter(DocumentChunk.patient_id == patient_id, DocumentChunk.embedding_model == embedding_model,
                    ClinicalDocument.status == "READY")
            .all()
        )
        if not rows:
            return []
        m = np.vstack([from_bytes(r.embedding) for r in rows])
        scores = m @ vector.astype(np.float32)
        order = np.argsort(-scores)[:k]
        return [Hit(rows[i].id, rows[i].document_id, float(scores[i])) for i in order]


class ChromaVectorIndex:
    name = "chroma"

    def __init__(self, path: str):
        import chromadb

        self.client = chromadb.PersistentClient(path=path)

    def _collection(self, embedding_model: str):
        safe = "chunks_" + "".join(c if c.isalnum() else "_" for c in embedding_model)[:50]
        # embeddings are always supplied by us, so Chroma never downloads its own default model
        return self.client.get_or_create_collection(safe, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def upsert(self, db: Session, chunks: Sequence[DocumentChunk]) -> None:
        by_model: dict = {}
        for c in chunks:
            by_model.setdefault(c.embedding_model, []).append(c)
        for model, group in by_model.items():
            self._collection(model).upsert(
                ids=[c.id for c in group],
                embeddings=[from_bytes(c.embedding).tolist() for c in group],
                documents=[c.text for c in group],
                metadatas=[{
                    "patient_id": c.patient_id, "document_id": c.document_id, "version_id": c.version_id,
                    "doc_type": c.doc_type, "doc_date": c.doc_date.isoformat() if c.doc_date else "",
                    "section": c.section or "", "page_start": c.page_start or 0, "page_end": c.page_end or 0,
                    "chunk_index": c.chunk_index,
                } for c in group],
            )

    def delete_document(self, db: Session, document_id: str) -> None:
        for col in self.client.list_collections():
            name = col if isinstance(col, str) else col.name
            self.client.get_collection(name).delete(where={"document_id": document_id})

    def query(self, db: Session, patient_id: str, vector: np.ndarray, k: int, embedding_model: str) -> List[Hit]:
        col = self._collection(embedding_model)
        res = col.query(query_embeddings=[vector.tolist()], n_results=k, where={"patient_id": patient_id})
        ids, metas, dists = res["ids"][0], res["metadatas"][0], res["distances"][0]
        live = {d for (d,) in db.query(ClinicalDocument.id).filter(ClinicalDocument.status == "READY",
                                                                     ClinicalDocument.id.in_({m["document_id"] for m in metas}))}
        return [Hit(i, m["document_id"], 1.0 - float(d)) for i, m, d in zip(ids, metas, dists) if m["document_id"] in live]


@lru_cache(maxsize=2)
def get_index(name: Optional[str] = None) -> VectorIndex:
    name = name or settings.vector_backend
    if name == "chroma":
        return ChromaVectorIndex(settings.chroma_path)
    return SqlVectorIndex()
