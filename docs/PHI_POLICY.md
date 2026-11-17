# PHI policy: what may reach the LLM

**Status: demo / synthetic data only.** Every patient and document in this repository is synthetic. This
policy defines what the system *enforces* so that the same code is safe to point at real data later.

## Decision

| Question | Decision |
|---|---|
| May a **hosted** LLM (OpenAI, Groq, ...) receive patient data? | Only after a signed **BAA** with that provider. In `staging`/`production` the code **refuses to call a hosted provider** unless `LLM_HOSTED_BAA_CONFIRMED=true`. You set that flag only after the BAA exists. |
| What if there is no BAA? | Use a **local model**: set `LLM_BASE_URL=http://localhost:11434/v1` (Ollama / vLLM / llama.cpp). Local endpoints are never blocked. With neither, the pipeline runs on rules + ML only and reports `phi_policy_blocked`. |
| Is the app usable with no LLM? | Yes. Rules and ML still produce an ESI; the response says `degraded: ["phi_policy_blocked"]` and flags the visit for clinician review. |

## What is sent (the de-identified payload)

Sent: age in years (`90+` for ages over 89), sex, the vitals, the **scrubbed** chief complaint, and up to
~7 **scrubbed** history snippets labelled `D1`, `D2`, ... (never real document ids).

Never sent: patient name, MRN, date of birth, contact details, calendar dates (converted to "2.1 years
ago"), document ids (they embed the MRN), user identity, facility identity.

Enforced in `services/pii.py` and `services/triage_pipeline.py`; tests in `backend/tests/test_pipeline_safety.py`.

## Limits you must know

* The scrubber is regex-based. It is a floor, not a guarantee: free text can contain identifiers it does
  not recognise (a street name, a rare occupation). This is why a BAA/local model is required for real PHI
  even though the payload is de-identified. Phase 2 should evaluate Presidio/NER and measure leakage.
* Safe Harbor (45 CFR 164.514(b)) also restricts geographic units, ages over 89 and all date elements;
  the pipeline handles ages and dates but nothing here is a legal determination.

## Audit

The exact de-identified text sent to the provider, its SHA-256, provider, model, prompt version, attempts
and latency are stored in the visit's `AI_TRIAGE` audit entry (`raw_payload_snapshot.llm`), and are visible
to clinicians through `GET /api/visits/{id}/audit`.
