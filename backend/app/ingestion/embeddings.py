"""
Embedding models behind one interface. The model id is stored on every chunk, so changing EMBEDDING_MODEL
makes existing chunks stale and `reindex --stale` re-embeds exactly those.

  hash-v2                    deterministic feature-hashing of distinct non-stopword words (2048 dims). Offline,
                             tiny, LEXICAL (not semantic). Default so the system runs on Render's 512 MB free tier
                             and in CI without downloading a model. (hash-v1, 384 dims with bigrams, was retired:
                             collision noise was as large as the signal of one shared word.)
  fastembed:<model>          ONNX sentence embeddings via `pip install fastembed`, e.g.
                             fastembed:BAAI/bge-small-en-v1.5 (384 dims, ~130 MB). General-domain; a clinical
                             model (e.g. a MedEmbed/BioLORD variant) must be chosen by retrieval evaluation in Phase 4.

What gets embedded: "<doc type> | <section> | <chunk text>" so the section heading travels with the chunk.
"""
import hashlib
import re
from functools import lru_cache
from typing import List, Protocol

import numpy as np

from backend.app.config import settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalised."""
        ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


_STOP = frozenset("a an and are as at be by for from has have in is it of on or the to was were with without patient".split())


class HashingEmbedder:
    name = "hash-v2"
    dim = 2048
    _TOKEN = re.compile(r"[a-z][a-z0-9-]+|\d+(?:\.\d+)?")

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            feats = {t for t in self._TOKEN.findall(text.lower()) if t not in _STOP}
            for f in feats:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalise(out)


class FastEmbedEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding  # optional dependency

        from backend.app.retrieval.models import verify_model

        self._model = TextEmbedding(model_name=model, cache_dir=settings.model_cache_dir)
        verify_model(model)  # refuses weights that differ from models.lock.json
        self.name = f"fastembed:{model}"
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed(self, texts: List[str]) -> np.ndarray:
        return _normalise(np.array(list(self._model.embed(texts)), dtype=np.float32))


@lru_cache(maxsize=2)
def get_embedder(name: str = "") -> Embedder:
    name = name or settings.embedding_model
    if name == "hash-v2":
        return HashingEmbedder()
    if name.startswith("fastembed:"):
        return FastEmbedEmbedder(name.split(":", 1)[1])
    raise ValueError(f"Unknown EMBEDDING_MODEL {name!r}")


def embedding_text(doc_type: str, section: str, text: str) -> str:
    return f"{doc_type.replace('_', ' ')} | {section or 'document'} | {text}"


def to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
