# TriageFlow production runbook

How to recognise, diagnose and fix the failures this system has actually had. Every incident below was either
**hit for real** (CI history) or **triggered on purpose in a drill** against the real app code, with the
commands and output kept so it can be repeated. Nothing here is hypothetical.

| # | Incident | How it happened | Severity |
|---|---|---|---|
| 1 | [CI red on every push: golden test aged out of its own rule](#incident-1-ci-red-on-every-push-golden-test-aged-out-of-its-own-rule) | Hit: CI runs #1-#5, 17-18 Nov 2026 | Deploys blocked |
| 2 | [Service refuses to start after deploy](#incident-2-service-refuses-to-start-after-deploy) | Drill: production config without its secrets | Outage |
| 3 | [LLM provider outage: every triage slowed to 5 s](#incident-3-llm-provider-outage-every-triage-slowed-to-5-s) | Drill: LLM endpoint taken down | Degraded |
| 4 | [Waiting room slow under production-sized history](#incident-4-waiting-room-slow-under-production-sized-history) | Drill: 300k visits in Postgres | Degraded |

**First five minutes, any incident.**
1. Check `GET /api/health`. Is the process up and the database reachable?
2. Check the platform logs (Render → service → Logs). Look for `Refusing to start`, `LLM circuit OPEN`,
   `Retrieval failed`, `Ingestion worker error`.
3. Check `GET /api/ai/metrics` (admin). It shows LLM failures by reason and per-provider circuit state.
4. Check whether triage responses carry a non-empty `degraded` list. Triage is designed to keep answering with
   the deterministic safety rules and the ML model when the LLM or retrieval fails, and that list says what is
   missing.

---

## Incident 1: CI red on every push, golden test aged out of its own rule

**How it happened.** For real. GitHub Actions CI #1-#5 on `main`, 17 Nov 23:23 to 18 Nov 00:26 IST, commits
`4363eb6` to `9377566`. The Render deploy of the same commits also failed.

**Symptoms.**
- The `backend` job failed at `python -m pytest` on every push, including README-only commits. The `frontend`
  job was green.
- None of the code under test had changed between a passing state and a failing one.

**Diagnosis.**
1. Reproduced locally with `python -m pytest -x`. Two independent failures turned up.
2. `test_auth.py::test_protected_environments_refuse_weak_secrets` expected an error mentioning `JWT_SECRET`.
   `ensure_secrets()` ran the storage checks first and raised `Refusing to start: SQLite is not a production
   database ...` before it ever looked at the JWT. This was an ordering bug, so it failed deterministically.
3. `test_golden.py::test_golden_scenario[PED-001]` failed with
   `required safety rules did not fire: {'R-PED-FEVER'}`. The rule (`age_years <= 0.25` and temp >= 38.0) and
   the scenario (age 0.25, 38.8 C) both looked right.
4. **Root cause of 3.** The triage endpoint uses the **stored patient's** age, not the request's. The seed
   computes that age from the date of birth (2026-08-15) and **today's date**. From 14 Nov 2026 onward the
   infant is 0.26 years old and no longer qualifies as "3 months or younger". The test was a time bomb that
   went off three days before the first push.

**Fix.**
- `config.ensure_secrets()` now collects the JWT problem together with the storage problems, so one error lists
  everything that is wrong.
- The test seed is pinned to a fixed date: `seed(..., today=date(2026, 10, 31))` in `tests/conftest.py`.
- The full suite passed locally afterwards (295 tests).

**Prevention.**
- Any test fixture that derives a value from "today" must pin the date. Clinical rules with age thresholds are
  exactly where this bites.
- A useful habit: when CI goes red on a commit that only touched docs, suspect time, the environment or
  dependency drift before code.

---

## Incident 2: Service refuses to start after deploy

**How it happened.** Drill: started the real app with `ENVIRONMENT=production` and only the development
defaults set. This is the most likely cause of a Render deploy that builds but never becomes healthy.

**Symptoms.**
- The deploy builds, then the service crash-loops and the health check never passes.
- The frontend shows network errors or the login page fails to log in.

**Diagnosis.** The startup log has a single line that lists every blocker:

```
RuntimeError: Refusing to start: JWT_SECRET must be set to at least 32 random characters in staging/production.
SQLite is not a production database; set DATABASE_URL to Postgres. ENCRYPTION_KEY is required (stored originals
and backups are encrypted with it). CORS_ORIGIN_REGEX accepts any subdomain; set CORS_ORIGINS to your frontend
URL and CORS_ORIGIN_REGEX=.
```

The refusal is deliberate (`config.ensure_secrets()` together with `ops/security.production_checks()`).
Starting a patient-data service with a guessable token secret or unencrypted storage would be worse than not
starting.

**Fix.** Set these in Render → service → **Environment**, then redeploy:

| Variable | Value |
|---|---|
| `JWT_SECRET` | 32+ random characters (`python -c "import secrets; print(secrets.token_urlsafe(48))"`) |
| `DATABASE_URL` | Postgres URL ending in `?sslmode=require` (for example Neon free tier) |
| `ENCRYPTION_KEY` | as described in `backend/.env.example` |
| `CORS_ORIGINS` | your frontend URL; set `CORS_ORIGIN_REGEX` to empty |
| `DEMO_USER_PASSWORD` | 12+ characters, otherwise no demo logins are created in production |

**Prevention.**
- Keep the list in `backend/.env.example` current.
- The error is designed to list all problems at once, so a deploy needs one fix-and-redeploy cycle, not one per
  missing variable.

---

## Incident 3: LLM provider outage, every triage slowed to 5 s

**How it happened.** Drill: pointed the LLM at a dead endpoint (`LLM_BASE_URL=http://127.0.0.1:9/v1`) and sent
real triage requests through the app (`TestClient`, SQLite, seeded users).

**Symptoms.**
- Triage still returned **HTTP 200 with the correct ESI 2** for a chest-pain case. The deterministic safety
  rules and the ML model don't need the LLM.
- Every response carried `degraded: ["llm_unavailable"]`.
- **Every** request took about 5 s instead of the usual 143 ms. During a real outage that latency lands on
  every nurse at once.

**Diagnosis.**

| Scenario (same request, warm process) | Latency |
|---|---|
| LLM disabled (baseline) | 143 ms |
| LLM endpoint down | **5,008 ms** |

- **Logs.** Each request logged `LLM unavailable after 2 attempt(s): ['local:APIConnectionError', ...]`.
- **Not the SDK.** Its own retries were already off (`max_retries=0`), so the retry budget was the app's:
  2 attempts, each a refused connection (about 2 s on Windows), plus 0.5 s of backoff.
- **Root cause.** Nothing *remembered* that the provider was down, so every request paid the full retry budget
  again.
- **Metrics.** `GET /api/ai/metrics` showed `failure_reasons: {'local:APIConnectionError': N}` growing one per
  request.

**Fix.** A per-provider **circuit breaker** in `llm/assessor.py`:
- **Opening.** After `LLM_CIRCUIT_FAILURE_THRESHOLD` (default 3) consecutive failed calls, the provider is
  skipped for `LLM_CIRCUIT_COOLDOWN_SECONDS` (default 30).
- **While open.** Triage goes straight to the next provider (the configured fallback, such as Groq) or returns
  the rules and ML result immediately, still marked `degraded`.
- **After the cooldown (half-open).** One trial call goes through. Success closes the circuit; failure opens it
  again.
- **Not counted.** Bad JSON from a *reachable* provider doesn't trip it; the JSON-repair retry handles that.
- **Tests.** Four cases in `test_llm_and_fusion.py` cover opening, skipping, fallback-while-open, half-open
  recovery and the bad-JSON exclusion.

Re-running the same drill with the fix:

```
warm1: HTTP 200 in 4766 ms | final ESI 2 | degraded=['llm_unavailable']
warm2: HTTP 200 in 4849 ms | final ESI 2 | degraded=['llm_unavailable']
ERROR  LLM circuit OPEN for local:llama3 after 3 consecutive failures; skipping it for 30s.
warm3: HTTP 200 in  166 ms | final ESI 2 | degraded=['llm_unavailable']
warm4: HTTP 200 in  121 ms | final ESI 2 | degraded=['llm_unavailable']
warm5: HTTP 200 in  128 ms | final ESI 2 | degraded=['llm_unavailable']
metrics: 'circuits': {'local:llama3': {'open': True, 'consecutive_failures': 3}}
```

**Operating it.**
- **Alert on the breaker.** Alert on the `LLM circuit OPEN` log line, or on `circuits.*.open` in
  `/api/ai/metrics`.
- **Configure a fallback.** Set `LLM_FALLBACK_API_KEY` / `LLM_FALLBACK_MODEL` / `LLM_FALLBACK_BASE_URL` so an
  open primary fails over to a second provider instead of to rules and ML only.
- **Protected environments.** In staging and production, hosted providers are blocked unless
  `LLM_HOSTED_BAA_CONFIRMED=true` (PHI policy). A "fallback never used" report there is usually that, not the
  breaker.

**Related observation: cold start.** The first triage after a process start took 19-74 s across these drills,
whatever the LLM state, because the ML, retrieval and OCR models load lazily. Free-tier hosts that sleep when
idle (Render free) hit this on the first request after every wake. Mitigation: warm the models in the startup
hook, or have a health check exercise a triage path, before taking traffic.

---

## Incident 4: Waiting room slow under production-sized history

**How it happened.** Drill: Postgres 16 with 300k historical visits and 1M historical documents, the app's real
queries replayed, `pg_stat_statements` on. The full write-up is in [sql-tuning.md](sql-tuning.md).

**Symptoms.**
- **Slow queue.** The waiting room (`GET /api/queue`), which dashboards poll, took about 0.8 s per poll.
- **CPU.** Postgres CPU climbed with the amount of history even though only about 300 patients were ever
  waiting.

**Diagnosis.**
1. **Find the slow statements.** `pg_stat_statements`, ordered by `total_exec_time`, showed the three slowest
   statements were all on `visits`: the queue, the deterioration-monitor tick and the retention check. Each had
   a mean of 77-89 ms and read about 6,400 blocks per call, which is the whole table.
2. **Confirm the plan.** `EXPLAIN (ANALYZE, BUFFERS)` showed `Parallel Seq Scan on visits ... Rows Removed by
   Filter: 99925` (x3 workers). All three ask only for active visits, which are 0.1% of the table, and there was
   no index for that.
3. **Count the queries.** Counting the SQL statements one queue request issued gave **225 statements for 225
   rows**. The route JOINed `patients` but lazy-loaded each `v.patient` separately (N+1).

**Fix.**
- **Indexes.** Two **partial indexes** over only the active rows (migration `0004`), 32 kB each against 14 MB
  for a full composite index.
- **Statistics.** `ANALYZE visits` after the build.
- **Query rewrite.** `contains_eager(Visit.patient)` in the queue and archive routes.
- **Result.** All three plans became Index Scans: 89 / 87 / 77 ms down to 0.05 / 6.1 / 2.8 ms. A queue poll
  went from 225 statements in 808 ms to 1 statement in 25 ms.
- **Tests.** `test_sql_tuning.py` guards both the index predicate and the N+1.

**Doing it on a live database.**
- **Build without blocking writes.** Use `CREATE INDEX CONCURRENTLY` so the build doesn't block inserts during
  triage.
- **Refresh statistics.** Run `ANALYZE` right after.
- **Confirm in production.** Check `pg_stat_user_indexes.idx_scan` to confirm the new indexes are used, and
  re-run the `pg_stat_statements` top-N query.

**Side issue hit during the drill.**
- **Symptom.** `VACUUM` in the Postgres container failed with
  `could not resize shared memory segment ... No space left on device`.
- **Cause.** Docker's default 64 MB `/dev/shm` is too small for parallel workers on a 721 MB table.
- **Fix.** Start the container with `--shm-size=1g`. Reducing `max_parallel_*` would have hidden the problem
  and changed the very plans being measured.

---

## Reproducing the drills

| Incident | Command |
|---|---|
| 1 | `git stash` the `conftest.py` date pin, then `python -m pytest backend/tests/test_golden.py` (fails on or after 14 Nov 2026) |
| 2 | `ENVIRONMENT=production DATABASE_URL=sqlite:///./x.db python -c "from fastapi.testclient import TestClient; from backend.app.main import app; TestClient(app).__enter__()"` |
| 3 | Run the API with `OPENAI_API_KEY=dummy LLM_BASE_URL=http://127.0.0.1:9/v1 LLM_MODEL=llama3 LLM_TIMEOUT_SECONDS=3`, send 5 triage requests, then read `GET /api/ai/metrics` |
| 4 | See [sql-tuning.md, section 7](sql-tuning.md#7-reproduce) |
