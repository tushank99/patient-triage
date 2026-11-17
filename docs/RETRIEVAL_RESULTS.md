# Retrieval benchmark (v1.0)

Synthetic set: 84 answerable + 36 unanswerable queries over 12 patients (chunk-level labels by construction). Real set: 18 hand-labelled queries over the seeded patients (document level).

> Templated synthetic notes are cleaner than real charts: use these numbers to COMPARE configurations, not as accuracy claims.
> With 84 answerable queries, differences under ~0.03 nDCG are within noise.

## Decision (Phase 4)

| | choice | why |
|---|---|---|
| Embedding model | **`minishlab/potion-retrieval-32M`** (fastembed 0.9.1, files pinned by sha256) | Best of five candidates on both sets (hybrid+expansion nDCG@10 0.895 synthetic, 0.979 real) and best fallback separation (AUROC 0.85). Static embeddings: ~4 ms per query, no transformer inference. Measured +~196 MB RAM, so it fits Render's 512 MB with the backend. Earlier full comparison: all-MiniLM-L6-v2 0.891, bge-small-en-v1.5 0.881 (and ~40x slower here), arctic-embed-xs 0.735. |
| Mode | **hybrid (dense + BM25, RRF) with query expansion** | Expansion is the biggest single win (baseline vector 0.808 -> 0.912 with potion; 0.338 -> 0.662 with the lexical embedder). Hybrid matches vector+expansion within noise on these sets and keeps exact-term matching (drug names, numbers, lab values) that static embeddings can blur. |
| Cross-encoder reranker | **off** | `ms-marco-MiniLM-L-6-v2` LOWERS nDCG for every real embedder (potion 0.895 -> 0.834) while adding ~700 ms p95 and ~170 MB. It is trained on web search, not lay-language clinical queries. It only helps the offline lexical embedder. Revisit with a clinical reranker. |
| Low-confidence gate | threshold **0.183** (cosine) for the default config | Keeps 90% of answerable queries and rejects 50% of unanswerable ones. For the offline `hash-v2` embedder the calibrated threshold is 0 (it cannot separate the two), i.e. the gate is effectively off there. |

Earlier five-model run (same benchmark, hybrid+expansion nDCG@10 synthetic / real): hash-v2 0.690 / 0.921 · arctic-embed-xs 0.735 / 0.936 ·
all-MiniLM-L6-v2 0.891 / 0.968 · bge-small-en-v1.5 0.881 / 0.960 · potion-retrieval-32M 0.895 / 0.979.

## Embedder `hash-v2`

| config | S@1 | S@3 | Recall@5 | MRR@10 | nDCG@10 | real S@3 | real MRR | real nDCG | fallback AUROC | unanswerable rejected @90% kept | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vector (baseline) | 0.190 | 0.333 | 0.440 | 0.287 | 0.338 | 0.944 | 0.755 | 0.808 | 0.463 | 0.000 | 4 |
| vector+expansion | 0.560 | 0.679 | 0.762 | 0.630 | 0.662 | 1.000 | 0.889 | 0.914 | 0.621 | 0.000 | 3 |
| bm25 | 0.286 | 0.298 | 0.321 | 0.297 | 0.303 | 0.556 | 0.556 | 0.470 | 0.599 | 0.000 | 2 |
| bm25+expansion | 0.631 | 0.667 | 0.690 | 0.652 | 0.662 | 0.778 | 0.778 | 0.735 | 0.767 | 0.000 | 2 |
| hybrid | 0.262 | 0.393 | 0.488 | 0.349 | 0.396 | 0.944 | 0.755 | 0.808 | 0.500 | 0.000 | 3 |
| hybrid+expansion | 0.607 | 0.702 | 0.774 | 0.663 | 0.690 | 1.000 | 0.889 | 0.921 | 0.648 | 0.000 | 3 |
| hybrid+expansion+rerank | 0.750 | 0.774 | 0.857 | 0.784 | 0.810 | 1.000 | 0.944 | 0.957 | 0.846 | 0.306 | 568 |

## Embedder `fastembed:minishlab/potion-retrieval-32M`

| config | S@1 | S@3 | Recall@5 | MRR@10 | nDCG@10 | real S@3 | real MRR | real nDCG | fallback AUROC | unanswerable rejected @90% kept | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vector (baseline) | 0.655 | 0.833 | 0.905 | 0.758 | 0.808 | 1.000 | 0.944 | 0.947 | 0.794 | 0.528 | 5 |
| vector+expansion | 0.845 | 0.940 | 0.952 | 0.894 | 0.912 | 1.000 | 0.972 | 0.979 | 0.855 | 0.472 | 3 |
| bm25 | 0.286 | 0.298 | 0.321 | 0.297 | 0.303 | 0.556 | 0.556 | 0.470 | 0.599 | 0.000 | 3 |
| bm25+expansion | 0.631 | 0.667 | 0.690 | 0.652 | 0.662 | 0.778 | 0.778 | 0.735 | 0.767 | 0.000 | 4 |
| hybrid | 0.607 | 0.810 | 0.893 | 0.722 | 0.780 | 1.000 | 0.944 | 0.947 | 0.778 | 0.444 | 4 |
| hybrid+expansion | 0.810 | 0.929 | 0.940 | 0.872 | 0.895 | 1.000 | 0.972 | 0.979 | 0.854 | 0.500 | 6 |
| hybrid+expansion+rerank | 0.762 | 0.798 | 0.869 | 0.801 | 0.834 | 1.000 | 0.944 | 0.957 | 0.847 | 0.278 | 671 |
