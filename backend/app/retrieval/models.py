"""
Pinned models. fastembed downloads ONNX weights from the Hugging Face hub; a hub repo can change under the
same name, so every model file we use is recorded with its sha256 in models.lock.json and checked on load.
A mismatch refuses to load (unless ALLOW_UNPINNED_MODELS=true): silently different embeddings would make every
stored vector incomparable with new queries.

  python -m backend.app.retrieval.models --prefetch            download configured models into MODEL_CACHE_DIR
  python -m backend.app.retrieval.models --prefetch --write-lock   ... and record their hashes (do this deliberately)

On Render, run --prefetch in the BUILD command: build output is kept, the runtime disk is not, so the models are
then present on every cold start without a download.
"""
import argparse
import hashlib
import json
import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from backend.app.config import settings

logger = logging.getLogger(__name__)
LOCK_PATH = os.path.join(os.path.dirname(__file__), "models.lock.json")
_HASHED = (".onnx", ".json", ".txt", ".model")


class ModelPinError(RuntimeError):
    pass


def _model_files(cache_dir: str, model: str) -> Dict[str, str]:
    """{relative path: sha256} for files belonging to `model` inside cache_dir."""
    tag = model.split("/")[-1].lower()
    out: Dict[str, str] = {}
    for root, _dirs, files in os.walk(cache_dir):
        if tag not in root.lower().replace("\\", "/"):
            continue
        for f in files:
            if f.endswith(_HASHED) and not f.startswith("."):
                path = os.path.join(root, f)
                if os.path.islink(path):
                    path = os.path.realpath(path)
                rel = os.path.relpath(os.path.join(root, f), cache_dir).replace("\\", "/")
                rel = "/".join(p for p in rel.split("/") if not (len(p) == 40 and all(c in "0123456789abcdef" for c in p)))  # drop snapshot hashes
                with open(path, "rb") as fh:
                    out[rel] = hashlib.sha256(fh.read()).hexdigest()
    return out


def load_lock(path: str = LOCK_PATH) -> Dict:
    if not os.path.exists(path):
        return {"models": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_model(model: str, cache_dir: Optional[str] = None, lock: Optional[Dict] = None) -> None:
    cache_dir = cache_dir or settings.model_cache_dir
    lock = load_lock() if lock is None else lock
    pinned = lock.get("models", {}).get(model)
    if pinned is None:
        if settings.allow_unpinned_models:
            logger.warning("Model %s is not in models.lock.json (allowed by ALLOW_UNPINNED_MODELS).", model)
            return
        raise ModelPinError(f"{model} is not pinned in models.lock.json. Run `python -m backend.app.retrieval.models --prefetch --write-lock`.")
    actual = _model_files(cache_dir, model)
    bad = {f: h for f, h in pinned["files"].items() if actual.get(f) != h}
    if bad and not settings.allow_unpinned_models:
        raise ModelPinError(f"{model}: files differ from the pinned hashes: {sorted(bad)}")


class CrossEncoderReranker:
    """fastembed cross-encoder. Returns raw relevance logits; confidence = sigmoid(logit)."""

    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._m = TextCrossEncoder(model_name=model, cache_dir=settings.model_cache_dir)
        verify_model(model)
        self.name = f"fastembed-ce:{model}"

    def score(self, query: str, texts: List[str]) -> np.ndarray:
        return np.array(list(self._m.rerank(query, texts)), dtype=np.float32)


@lru_cache(maxsize=2)
def get_reranker(name: Optional[str] = None):
    name = settings.reranker_model if name is None else name
    return CrossEncoderReranker(name) if name else None


def configured_models() -> List[str]:
    models = []
    if settings.embedding_model.startswith("fastembed:"):
        models.append(settings.embedding_model.split(":", 1)[1])
    if settings.reranker_model:
        models.append(settings.reranker_model)
    return models


def prefetch(write_lock: bool = False, extra: Optional[List[str]] = None) -> Dict:
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    lock = load_lock()
    names = configured_models() + list(extra or [])
    rerankers = {m["model"] for m in TextCrossEncoder.list_supported_models()}
    for name in names:
        (TextCrossEncoder if name in rerankers else TextEmbedding)(model_name=name, cache_dir=settings.model_cache_dir)
        files = _model_files(settings.model_cache_dir, name)
        if write_lock:
            import fastembed

            lock.setdefault("models", {})[name] = {"fastembed": fastembed.__version__, "files": files}
            print(f"pinned {name}: {len(files)} files")
        else:
            verify_model(name, lock=lock)
            print(f"verified {name}")
    if write_lock:
        with open(LOCK_PATH, "w", encoding="utf-8") as f:
            json.dump(lock, f, indent=2, sort_keys=True)
    return lock


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    ap.add_argument("--model", action="append", default=[], help="extra model to fetch (repeatable)")
    a = ap.parse_args()
    if a.prefetch:
        prefetch(a.write_lock, a.model)
