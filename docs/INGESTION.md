# Document ingestion (Phase 3)

```
upload ──► validate (bytes, size, safety, optional ClamAV) ──► store original ──► queue INGEST job
                                                                                   │
worker:  parse (PyMuPDF / python-docx / text) ──► section-aware chunks ──► embed ──► replace chunks + index
```

## Where things live
| What | Where | Persistent on Render free? |
|---|---|---|
| documents, versions, chunk text, **vectors**, jobs | main database (`clinical_documents`, `document_versions`, `document_chunks`, `ingestion_jobs`) | **yes** (Neon) |
| original files | `STORAGE_BACKEND=local` → `backend/data/uploads/` · `s3` → any S3-compatible bucket | local: **no** (wiped on redeploy) · s3: yes |
| optional Chroma index | `VECTOR_BACKEND=chroma`, `CHROMA_PATH` | **no** without a paid disk |

The database is the source of truth. Chroma (if enabled) is a derived index rebuilt with `POST /api/ingestion/reindex {"scope":"all"}`.
Losing local originals on Render only matters for re-*parsing*; re-chunking and re-embedding use the stored
`parsed_blocks`, so search keeps working. For durable originals use Cloudflare R2 or Backblaze B2 (S3 API, free tiers):
`STORAGE_BACKEND=s3 S3_BUCKET=... S3_ENDPOINT_URL=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...` and `pip install boto3`.

## Upload validation (`ingestion/validation.py`)
Type from magic bytes (never the filename/Content-Type). Accepted: PDF, DOCX, UTF-8 text, up to `MAX_UPLOAD_MB` (10).
Rejected: encrypted PDFs, PDFs with JavaScript/launch actions/embedded files, > `MAX_PDF_PAGES`, macro-enabled or
zip-bomb DOCX, binary data posing as text. Malware scanning runs when `CLAMAV_HOST` is set (fails closed if the
scanner is down); otherwise each version records `scan_result = not_scanned`. Image-only PDFs fail with "OCR arrives in Phase 5".

## Chunking (`ingestion/chunker.py`)
Sections start at headings (PDF font size/bold, DOCX heading styles, text heading patterns, plus a clinical
heading lexicon). Chunks of ~`CHUNK_TARGET_CHARS` (900) never cross a section, overlap by ~`CHUNK_OVERLAP_CHARS`
(150, whole sentences), and keep section, page range and character offsets. What is embedded:
`"<doc type> | <section> | <text>"`.

## Embeddings
`EMBEDDING_MODEL=hash-v2` (default): offline lexical feature hashing, so it fits a 512 MB instance and CI. It is
**not semantic**. `fastembed:BAAI/bge-small-en-v1.5` (`pip install fastembed`) gives real sentence embeddings;
which (clinical) model to use is decided by the retrieval benchmark in Phase 4.

## Versioning, dedupe, re-index, deletion
* Same bytes for the same patient → `duplicate` (no new job).
* `POST /api/documents/{id}/versions` → version N+1; version N stays searchable until N+1 is indexed.
* Each version records `chunker` and `embedding_model`. Change either in config, then
  `POST /api/ingestion/reindex {"scope":"stale"}` re-processes only stale versions, without re-parsing.
* `DELETE /api/documents/{id}` (charge nurse, physician, admin) removes chunks from retrieval in the same
  transaction; a DELETE job then purges stored originals, extracted text and any Chroma entries, with retries.
  A metadata tombstone remains (who, when, checksums). Anonymising a patient purges everything, no tombstone.

## Jobs (`ingestion/worker.py`)
DB-backed queue: no Redis needed. A conditional UPDATE claims a job, so several workers can run safely.
Transient errors retry with exponential backoff (5 s, 10 s, ... up to 10 min) up to `INGESTION_MAX_ATTEMPTS`
(3), then **DEAD** (dead letter). Permanent errors (corrupt or unparseable file, missing original) go to **FAILED** at once.
Jobs left RUNNING past `INGESTION_LEASE_MINUTES` (a crashed worker) are requeued.
In-process loop by default (`INGESTION_WORKER_ENABLED`); or run `python -m backend.app.ingestion.worker` separately.
Monitoring (charge nurse/admin): `GET /api/ingestion/stats` (jobs by status, per-stage timings, stale versions,
backends), `GET /api/ingestion/jobs?status=DEAD`, `POST /api/ingestion/jobs/{id}/retry`.

## Migrations
Alembic (`backend/migrations`). The app runs `upgrade head` at startup; a pre-Alembic database is stamped at the
baseline first, so existing data survives. Create a migration after changing models:
`alembic -c backend/alembic.ini revision --autogenerate -m "..."`. A test fails if models and migrations diverge.

## Licence note
PyMuPDF is **AGPL-3.0**. Fine for this open demo; a closed-source deployment needs Artifex's commercial licence
or a swap to `pypdf` (BSD) behind the same `parse_pdf` function.
