# Retrieval (Phase 4)

```
complaint ─► normalise + expand ─► dense (multi-query) ┐
                                   BM25 (global IDF)   ┴─► RRF ─► [cross-encoder rerank] ─► low-confidence gate ─► parent section, compressed
```
Code: `backend/app/retrieval/`. Benchmark: `backend/retrieval_eval/`. Results: `docs/RETRIEVAL_RESULTS.md`.

## Components
* **Query normalisation** (`normalize.py`, `medical_lexicon.json`): whole-token abbreviation expansion (afib, SOB, MI, HTN,
  CKD, ...) and lay-term mapping ("blood thinner" → anticoagulant, warfarin, apixaban ...). Produces multi-query variants
  (original, abbreviations spelled out, clinical terms only). Expansion terms count 0.6 in BM25; the nurse's own words count 1.
  Ambiguous abbreviations (MS, PT, CA) are deliberately absent. The lexicon was written by an engineer: have a clinician review it.
* **Dense**: every variant is embedded; a chunk's score is its best cosine across variants.
* **BM25** (`bm25.py`): Okapi BM25 over the patient's chunks with document frequencies from the whole collection (per-patient
  IDF over ~20 chunks would be noise). Postgres full-text search could replace it later; it is not BM25, so it was not used.
* **Fusion**: Reciprocal Rank Fusion (k = 60).
* **Reranker** (optional, `RERANKER_MODEL`): fastembed cross-encoder over the top 20.
* **Low-confidence fallback**: the best passage's relevance (cross-encoder sigmoid, else dense cosine, else saturated BM25)
  is compared with a threshold **calibrated on the benchmark** (keep ≥ 90% of answerable queries; see `thresholds.json`).
  Below it, no narrative note is sent to the LLM (allergies, medications and problem list still are), the nurse sees
  "no note matched this complaint confidently", and the LLM is told records exist but none matched (so it does not
  assume an empty history). `has_history` and the ML feature are NOT affected.
* **Parent-child** : the hit chunk is returned with its whole section (from the stored extracted text, so overlap is not
  duplicated), compressed to `RETRIEVAL_CONTEXT_CHARS` around the hit by keeping the most query-relevant sentences in order.
  One passage per section.
* **Caching** (`cache.py`): query embeddings (model + exact text) and per-patient chunk matrices (patient id + a fingerprint
  that changes when that patient's chunks change). In-process, never persisted, never shared across patients.

## Hard patient filter
`retrieve(db, patient_id, ...)` is the only entry point and refuses a missing id. Every stage works on chunks loaded with
`WHERE patient_id = :id` (plus an assertion). `backend/tests/test_retrieval.py` attacks it: canary tokens from another
patient, SQL-like and wildcard queries, the other patient's MRN, an "ignore the patient filter" instruction inside a
document, a reranker that tries to promote foreign text (it never even receives it), alternating cached queries, unknown
patients and deleted documents. 4 configurations × 11 attacks, all must return only the requesting patient's chunks.

## Model choice and pinning
See the decision in `docs/RETRIEVAL_RESULTS.md`. Model files are pinned by sha256 in
`backend/app/retrieval/models.lock.json`; a changed file refuses to load. Prefetch at **build** time on Render:
`pip install -r backend/requirements.txt && python -m backend.app.retrieval.models --prefetch`.
