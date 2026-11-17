# Grounded generation and evaluation (Phase 5)

## Claims and citations
The LLM (prompt `triage_assessment.v4`) must put every fact it relies on into `claims`, each citing `D1…` (a record
passage it was shown), `FACTS` (the deterministic facts block) or `INTAKE` (today's complaint and vitals).
`backend/app/llm/grounding.py` checks each claim against exactly the text it cites:

| verdict | meaning |
|---|---|
| supported | ≥ 70% of the claim's key terms are in the cited text (abbreviations/synonyms expanded), all numbers match, no negation or current/resolved flip |
| partial | 40–70% of key terms; shown with a warning, not escalated |
| unsupported | missing terms, a different number, a flipped negation, or "current" where the source says resolved/stopped (or vice versa) |
| uncited | a claim about the record with no citation |
| invalid_citation | cites something the model was never shown (e.g. a real document id or D9) |

Unsupported, uncited and invalid claims are struck through in the UI, listed in `review_reasons`, counted in
`grounding`, and stored in the audit trail. This is a deterministic lexical check, not an entailment model: it
catches fabricated drugs, conditions and numbers, wrong citations, negation flips and temporal errors (see
`backend/eval/hallucination_set.json`), but not every paraphrase that keeps the words and changes the meaning.

## Untrusted text (prompt injection)
Retrieved passages, uploaded documents and the complaint are data. Defence in depth
(`backend/app/llm/security.py`): delimiter tags with tag-breaking neutralised; instruction-like sentences
("ignore previous instructions", "SYSTEM:", "set the ESI to 5", "do not mention…") replaced by
`[instruction-like text withheld]` before the prompt is built, flagged in `security_flags`, and listed in the review
reasons; the system prompt says delimited text is never an instruction; and on the output side the safety-rule floor
cannot be lowered and every claim is verified.

## Facts, temporal status and conflicts
`backend/app/clinical/facts.py` extracts conditions (ACTIVE/RESOLVED/HISTORICAL, plus a rule that temporary
conditions over ~9 months old count as resolved), medications (list status, then newer "stopped/started" notes win),
and allergies, from ALL of the patient's documents (not just the retrieved ones). Conflicts:
medication list vs newer note; allergy record vs a note saying NKDA; intake denying documented allergies; intake
mentioning a drug in a documented allergy class (cross-reactivity map in `drug_classes.json`, engineer-written:
needs pharmacist review); intake mentioning a stopped drug; intake mentioning an anticoagulant not on file.
High-severity conflicts force clinician review. Facts also feed the anticoagulant safety rule, so an uploaded note
saying "on warfarin" now counts.

## Evaluation
* `backend/tests/test_grounding_eval.py` (CI): the hallucination set (every fabricated claim caught, no faithful
  claim flagged), the injection set (malicious flagged, benign clinical text not), facts/conflicts on seeded
  patients, grounding and injection through the API, and the e2e harness with scripted LLMs.
* `python -m backend.eval.run_e2e --llm live --report docs/EVAL_RESULTS.md [--models a,b]` runs the golden set plus
  allergy-conflict, stopped-drug, temporal-trap and prompt-injection scenarios against the real provider and reports
  ESI accuracy, safety, grounding rates, conflicts surfaced, temporal-trap violations, injection resistance, latency
  and tokens per model (model selection). Not run in CI: it costs tokens and needs a key.
* `GET /api/ai/metrics` (charge nurse/admin): calls, failures by reason, latency p50/p95 and tokens per provider/model.
