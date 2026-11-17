"""
Retrieval benchmark: plain-vector baseline vs BM25 vs hybrid vs +query expansion vs +cross-encoder rerank,
for one or more embedding models. Uses an ISOLATED temporary database and the real ingestion pipeline.

  python -m backend.retrieval_eval.run                                   # hash-v2 only (no downloads)
  python -m backend.retrieval_eval.run --embedders hash-v2,fastembed:snowflake/snowflake-arctic-embed-xs \
         --reranker Xenova/ms-marco-MiniLM-L-6-v2 --write-thresholds --report docs/RETRIEVAL_RESULTS.md

Metrics (synthetic set, chunk level): Success@1/3/5 (any relevant chunk in top k), Recall@5, MRR@10, nDCG@10.
Real set (seeded patients, hand-labelled, document level): Success@3, MRR@10, nDCG@10.
Fallback: AUROC of the top relevance score for answerable vs unanswerable queries, and the threshold that
keeps >= 90% of answerable queries, with the share of unanswerable queries it rejects.
"""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_TMP = tempfile.mkdtemp(prefix="retrieval-bench-")
os.environ.update({
    "DATABASE_URL": f"sqlite:///{Path(_TMP, 'bench.db').as_posix()}", "STORAGE_PATH": str(Path(_TMP, "uploads")),
    "SEED_ON_STARTUP": "false", "INGESTION_WORKER_ENABLED": "false", "LLM_ENABLED": "false", "ENVIRONMENT": "test",
    "ALLOW_UNPINNED_MODELS": "true",  # candidates are compared BEFORE one is pinned
    "VECTOR_BACKEND": "sql", "EMBEDDING_MODEL": "hash-v2",
})
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.app.config import settings  # noqa: E402
from backend.app.database import SessionLocal  # noqa: E402
from backend.app.ingestion import embeddings  # noqa: E402
from backend.app.ingestion.service import add_upload, enqueue_reindex  # noqa: E402
from backend.app.ingestion.worker import run_pending_jobs  # noqa: E402
from backend.app.models import DocumentChunk, Patient  # noqa: E402
from backend.app.retrieval import bm25  # noqa: E402
from backend.app.retrieval.cache import PATIENT_CHUNKS, QUERY_EMBEDDINGS  # noqa: E402
from backend.app.retrieval.retriever import THRESHOLDS_PATH, RetrievalConfig, retrieve, signature  # noqa: E402
from backend.retrieval_eval.corpus import BENCHMARK_VERSION, generate  # noqa: E402

REAL_QUERIES = Path(__file__).with_name("real_queries.json")


def configs(with_rerank: bool):
    base = dict(parent_child=False, top_k=10, rerank=False)
    out = [
        RetrievalConfig(name="vector (baseline)", dense=True, sparse=False, expand=False, **base),
        RetrievalConfig(name="vector+expansion", dense=True, sparse=False, expand=True, **base),
        RetrievalConfig(name="bm25", dense=False, sparse=True, expand=False, **base),
        RetrievalConfig(name="bm25+expansion", dense=False, sparse=True, expand=True, **base),
        RetrievalConfig(name="hybrid", dense=True, sparse=True, expand=False, **base),
        RetrievalConfig(name="hybrid+expansion", dense=True, sparse=True, expand=True, **base),
    ]
    if with_rerank:
        out.append(RetrievalConfig(name="hybrid+expansion+rerank", dense=True, sparse=True, expand=True, parent_child=False, top_k=10, rerank=True))
    return out


# ------------------------------------------------------------------ metrics


def success_at(ranked, rel, k):
    return float(any(r in rel for r in ranked[:k]))


def recall_at(ranked, rel, k):
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def mrr(ranked, rel, k=10):
    return next((1.0 / (i + 1) for i, r in enumerate(ranked[:k]) if r in rel), 0.0)


def ndcg(ranked, rel, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def pick_threshold(pos, neg, keep=0.90):
    """Highest threshold that still keeps `keep` of answerable queries above it."""
    if not pos:
        return None, None, None
    t = float(np.quantile(np.array(pos), 1 - keep))
    kept = sum(p >= t for p in pos) / len(pos)
    rejected = (sum(n < t for n in neg) / len(neg)) if neg else None
    return t, kept, rejected


# ------------------------------------------------------------------ setup


def build_db(patients) -> None:
    from backend.scripts.seed_data import seed

    seed(reset=True)  # migrations + the 36 seeded patients and their history, through the ingestion pipeline
    db = SessionLocal()
    for bp in patients:
        p = Patient(mrn=bp.mrn, name=f"Bench {bp.mrn}", age_years=bp.age, gender=bp.sex)
        db.add(p)
        db.flush()
        for d in bp.docs:
            add_upload(db, patient=p, data=d.text.encode(), doc_type=d.doc_type, title=d.title, doc_date=None, actor="BENCH",
                       document_id=d.id, source=f"retrieval-benchmark@{BENCHMARK_VERSION}", is_synthetic=True)
    db.commit()
    db.close()
    run_pending_jobs()


def use_embedder(name: str) -> None:
    settings.embedding_model = name
    embeddings.get_embedder.cache_clear()
    QUERY_EMBEDDINGS.clear()
    PATIENT_CHUNKS.clear()
    db = SessionLocal()
    enqueue_reindex(db, "stale", "BENCH")
    db.commit()
    db.close()
    run_pending_jobs()
    bm25.GLOBAL._checked = 0  # force a DF refresh


def evaluate(db, cfg, reranker, patients, queries, real) -> dict:
    ids = {p.mrn: p.id for p in db.query(Patient).filter(Patient.mrn.in_([q.mrn for q in queries] + [q["mrn"] for q in real]))}
    chunk_doc = dict(db.query(DocumentChunk.id, DocumentChunk.document_id))
    chunk_text = dict(db.query(DocumentChunk.id, DocumentChunk.text))
    rows, pos_rel, neg_rel, lat = [], [], [], []
    for q in queries:
        res = retrieve(db, ids[q.mrn], q.text, cfg, reranker=reranker)
        lat.append(res.timings_ms.get("total", 0.0))
        score = res.top_relevance if res.top_relevance is not None else -1.0
        if not q.positive:
            neg_rel.append(score)
            continue
        pos_rel.append(score)
        rel = {cid for cid in res.ranked_chunk_ids if q.relevant_text in chunk_text[cid]}
        rel |= {cid for cid, d in chunk_doc.items() if d.startswith(q.mrn + ":") and q.relevant_text in chunk_text[cid]}
        r = res.ranked_chunk_ids
        rows.append([success_at(r, rel, 1), success_at(r, rel, 3), success_at(r, rel, 5), recall_at(r, rel, 5), mrr(r, rel), ndcg(r, rel)])
    real_rows = []
    for q in real:
        res = retrieve(db, ids[q["mrn"]], q["text"], cfg, reranker=reranker)
        docs = list(dict.fromkeys(chunk_doc[c] for c in res.ranked_chunk_ids))
        rel = set(q["relevant"])
        real_rows.append([success_at(docs, rel, 1), success_at(docs, rel, 3), mrr(docs, rel), ndcg(docs, rel)])
    m = np.array(rows).mean(axis=0)
    rm = np.array(real_rows).mean(axis=0) if real_rows else [None] * 4
    t, kept, rejected = pick_threshold(pos_rel, neg_rel)
    return {
        "synthetic": {"n": len(rows), "success@1": m[0], "success@3": m[1], "success@5": m[2], "recall@5": m[3], "mrr@10": m[4], "ndcg@10": m[5]},
        "real": {"n": len(real_rows), "success@1": rm[0], "success@3": rm[1], "mrr@10": rm[2], "ndcg@10": rm[3]},
        "fallback": {"auroc": auroc(pos_rel, neg_rel), "threshold": t, "answerable_kept": kept, "unanswerable_rejected": rejected,
                     "n_unanswerable": len(neg_rel)},
        "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
        "signature": signature(cfg, getattr(reranker, "name", None)),
    }


def _r(x):
    return "n/a" if x is None else f"{x:.3f}"


def render(results: dict) -> str:
    lines = [f"# Retrieval benchmark (v{BENCHMARK_VERSION})", "",
             f"Synthetic set: {results['n_synthetic_positive']} answerable + {results['n_synthetic_negative']} unanswerable queries over "
             f"{results['n_bench_patients']} patients (chunk-level labels by construction). Real set: {results['n_real']} hand-labelled queries "
             "over the seeded patients (document level).", "",
             "> Templated synthetic notes are cleaner than real charts: use these numbers to COMPARE configurations, not as accuracy claims.", ""]
    for emb, rows in results["embedders"].items():
        lines += [f"## Embedder `{emb}`", "",
                  "| config | S@1 | S@3 | Recall@5 | MRR@10 | nDCG@10 | real S@3 | real MRR | real nDCG | fallback AUROC | unanswerable rejected @90% kept | p95 ms |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in rows.items():
            s, rl, f = r["synthetic"], r["real"], r["fallback"]
            lines.append(f"| {name} | {_r(s['success@1'])} | {_r(s['success@3'])} | {_r(s['recall@5'])} | {_r(s['mrr@10'])} | {_r(s['ndcg@10'])} | "
                         f"{_r(rl['success@3'])} | {_r(rl['mrr@10'])} | {_r(rl['ndcg@10'])} | {_r(f['auroc'])} | {_r(f['unanswerable_rejected'])} | {r['latency_ms']['p95']:.0f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedders", default="hash-v2")
    ap.add_argument("--reranker", default="")
    ap.add_argument("--out", default=str(ROOT / "backend" / "retrieval_eval" / "results.json"))
    ap.add_argument("--report", default="")
    ap.add_argument("--write-thresholds", action="store_true")
    ap.add_argument("--patients", type=int, default=12)
    a = ap.parse_args()

    patients, queries = generate(n_patients=a.patients)
    real = json.loads(REAL_QUERIES.read_text(encoding="utf-8"))["queries"]
    build_db(patients)
    reranker = None
    if a.reranker:
        from backend.app.retrieval.models import CrossEncoderReranker

        reranker = CrossEncoderReranker(a.reranker)

    results = {"benchmark_version": BENCHMARK_VERSION, "n_bench_patients": len(patients), "n_real": len(real),
               "n_synthetic_positive": sum(q.positive for q in queries), "n_synthetic_negative": sum(not q.positive for q in queries),
               "embedders": {}}
    thresholds = {}
    if os.path.exists(THRESHOLDS_PATH):
        thresholds = json.loads(Path(THRESHOLDS_PATH).read_text(encoding="utf-8"))
    for emb in [e.strip() for e in a.embedders.split(",") if e.strip()]:
        use_embedder(emb)
        db = SessionLocal()
        rows = {}
        for cfg in configs(reranker is not None):
            r = evaluate(db, cfg, reranker if cfg.rerank else None, patients, queries, real)
            rows[cfg.name] = r
            if r["fallback"]["threshold"] is not None:
                thresholds[r["signature"]] = {"threshold": round(r["fallback"]["threshold"], 4), "answerable_kept": r["fallback"]["answerable_kept"],
                                              "unanswerable_rejected": r["fallback"]["unanswerable_rejected"], "auroc": r["fallback"]["auroc"],
                                              "benchmark_version": BENCHMARK_VERSION}
            print(f"{emb:<48} {cfg.name:<26} nDCG@10 {r['synthetic']['ndcg@10']:.3f}  MRR {r['synthetic']['mrr@10']:.3f}  "
                  f"real nDCG {r['real']['ndcg@10']:.3f}  AUROC {_r(r['fallback']['auroc'])}  p95 {r['latency_ms']['p95']:.0f} ms")
        db.close()
        results["embedders"][emb] = rows
    Path(a.out).write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    if a.report:
        Path(a.report).write_text(render(results), encoding="utf-8")
    if a.write_thresholds:
        Path(THRESHOLDS_PATH).write_text(json.dumps(thresholds, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
