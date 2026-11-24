# SQL performance tuning: the live slice of `visits`

**Result:** the three slowest statements found by `pg_stat_statements` all went from a **Parallel Seq Scan** of
the whole `visits` table to an **Index Scan** on a 32 kB partial index. Their mean server time dropped from
**89 / 87 / 77 ms to 0.05 / 6.1 / 2.8 ms**, and the replayed workload spent 60.5 s in them before and 2.1 s after.
A query rewrite also removed an N+1 in the waiting room: one poll went from **225 SQL statements and 808 ms to
1 statement and 25 ms**.

| | |
|---|---|
| Database | PostgreSQL 16.16 (Docker, local), `pg_stat_statements`, `track_io_timing=on` |
| Schema | the app's own Alembic migrations (0001-0003), unchanged |
| Data | synthetic, production-shaped: 50k patients, 1M historical clinical documents (721 MB), 300k historical ED visits of which 300 are still active |
| Workload | `backend/scripts/sql_tuning/workload.py`: the app's own ORM queries, 300 rounds |
| Raw output | [`docs/sql-tuning/`](sql-tuning/) (traces and plans, verbatim) |

## 1. Trace: find the slowest statements

The workload replays what TriageFlow does against its history tables: the triage history lookup and document
count, the chart's document list, the upload duplicate check, the waiting-room poll, the deterioration-monitor
tick and the retention job's per-patient check. Then:

```sql
SELECT calls, total_exec_time, mean_exec_time, shared_blks_hit + shared_blks_read AS blocks, query
FROM pg_stat_statements ORDER BY total_exec_time DESC;
```

| calls | total ms | mean ms | blocks | statement (app code) |
|---:|---:|---:|---:|---|
| 300 | 26,818 | **89.39** | 1,932,300 | active visits for a patient (`ops/retention.py`) |
| 300 | 26,019 | **86.73** | 2,096,092 | waiting room for a facility (`routers/queue.py`) |
| 100 | 7,683 | **76.83** | 638,300 | `status = 'WAITING'` (`background_tasks.py`, monitor) |
| 300 | 185 | 0.61 | 7,039 | count READY documents for a patient (`history_store.py`) |
| 300 | 136 | 0.45 | 7,045 | document list for a patient (`routers/documents.py`) |

([full output](sql-tuning/01_pg_stat_statements_before.txt))

**Diagnosis.**
- **Not the problem: `clinical_documents`.** Its queries are already sub-millisecond on the server because
  `ix_clinical_documents_patient_id` serves them. The ~8 ms the client sees is round trip and ORM overhead, so
  there was nothing worth indexing there.
- **The problem: `visits`, the historical encounter table.** Each statement reads about 6,400 blocks per call,
  which is the whole table. All three only want the **live** rows (`status IN ('WAITING','NEEDS_REASSESSMENT')`):
  300 of 300,000, or 0.1%. Neither `status` nor `visits.patient_id` was indexed.

## 2. Before: `EXPLAIN (ANALYZE, BUFFERS)`

Each query runs once to warm the cache, then the warm run is measured
([script](../backend/scripts/sql_tuning/explain_queries.sql), [full output](sql-tuning/02_explain_before.txt)).

**Q1, retention check.**

```
Aggregate  (actual time=69.096..77.818 rows=1)  Buffers: shared hit=6441
  ->  Gather  Workers Launched: 2
        ->  Parallel Seq Scan on visits  (actual rows=0 loops=3)
              Filter: (status = ANY ('{WAITING,NEEDS_REASSESSMENT}') AND patient_id = 'pat-12345')
              Rows Removed by Filter: 100000
Execution Time: 77.869 ms
```

**Q2, waiting room.**

```
Gather Merge  (actual time=63.110..72.053 rows=225)  Buffers: shared hit=7093
  ->  Sort  Sort Key: final_esi_score, arrival_time
        ->  Nested Loop
              ->  Parallel Seq Scan on visits  Rows Removed by Filter: 99925 (x3)
              ->  Index Only Scan using patients_pkey on patients
Execution Time: 72.184 ms
```

**Q3, monitor tick.** Parallel Seq Scan on visits, Rows Removed by Filter: 99950 (x3), Buffers: shared hit=6383,
**Execution Time: 63.775 ms**.

## 3. Change: partial indexes, `ANALYZE`, one query rewrite

**Indexes** ([migration 0004](../backend/migrations/versions/0004_active_visit_partial_indexes.py), mirrored in `models.Visit.__table_args__`):

```sql
CREATE INDEX ix_visits_active_queue   ON visits (facility_id, final_esi_score, arrival_time)
  WHERE status IN ('WAITING', 'NEEDS_REASSESSMENT');
CREATE INDEX ix_visits_active_patient ON visits (patient_id)
  WHERE status IN ('WAITING', 'NEEDS_REASSESSMENT');
ANALYZE visits;  -- refresh planner statistics so the new indexes are costed on current data
```

**Why partial and composite.**
- **The predicate is the app's own definition of "active".** The index holds only live rows, so it stays tiny
  however much history accumulates: **32 kB each, against 14 MB** for the equivalent full composite index on
  `(status, facility_id, final_esi_score, arrival_time)`, which I built for comparison and dropped
  ([output](sql-tuning/03_index_change.txt)). Writes to historical rows never touch it.
- **The column order matches the queue query.** Equality on `facility_id` comes first, then the `ORDER BY`
  columns, so Postgres reads rows already sorted and the Sort node disappears.
- **Q3 needs no index of its own.** The planner proves that `status = 'WAITING'` implies the index predicate and
  reuses `ix_visits_active_queue`.
- **A test keeps the predicate in sync.** `test_sql_tuning.py` fails if `ACTIVE_STATUSES` changes without the
  index predicate changing too.

**Query rewrite (N+1).** `routers/queue.py` joined `patients` but did not *load* them through the join, so
building the response lazy-loaded each `v.patient` with its own `SELECT`. With the new index in place, one queue
poll through the real route function still issued **225 statements in 808 ms**. Adding
`.options(contains_eager(Visit.patient))` fills the relationship from the existing JOIN: **1 statement, 25 ms**.
The archive endpoint had the same pattern and got the same fix. `test_queue_loads_patients_in_the_same_query`
fails if the N+1 comes back (verified: it reports 3 per-row queries with the fix reverted).

**Not rewritten.** Q1's `.count()` wraps the query in `SELECT count(*) FROM (SELECT ...)`. The planner flattens
that subquery (there is no Subquery Scan node in either plan), and after the index the plan is an Index Only Scan
with 0 heap fetches, so a rewrite would gain nothing.

## 4. After: the plan change

[Full output](sql-tuning/04_explain_after.txt).

```
Q1  Aggregate (actual time=0.054..0.055)  Buffers: shared hit=2
      ->  Index Only Scan using ix_visits_active_patient on visits  Heap Fetches: 0
    Execution Time: 0.126 ms

Q2  Nested Loop (actual time=0.153..3.038 rows=225)  Buffers: shared hit=714
      ->  Index Scan using ix_visits_active_queue on visits  Buffers: shared hit=38   (no Sort node)
      ->  Index Only Scan using patients_pkey on patients
    Execution Time: 3.164 ms

Q3  Index Scan using ix_visits_active_queue on visits  Filter: status = 'WAITING'  Buffers: shared hit=67
    Execution Time: 0.345 ms
```

## 5. Results

| Query | Plan before | Plan after | EXPLAIN ms | Mean ms (`pg_stat_statements`, 300 calls) | Buffers / call |
|---|---|---|---:|---:|---:|
| Q1 retention check | Parallel Seq Scan | Index Only Scan (`ix_visits_active_patient`) | 77.87 → **0.13** | 89.39 → **0.05** | 6,441 → **2** |
| Q2 waiting room | Parallel Seq Scan + Sort | Index Scan (`ix_visits_active_queue`), no Sort | 72.18 → **3.16** | 86.73 → **6.09** | 7,093 → **714** |
| Q3 monitor tick | Parallel Seq Scan | Index Scan (`ix_visits_active_queue`) | 63.78 → **0.35** | 76.83 → **2.84** | 6,383 → **67** |
| Queue poll, end to end | 225 statements | 1 statement | | 808 → **25** (client-side, includes Python) | |

The workload spent 60,520 ms in the three statements before and 2,127 ms after
([before](sql-tuning/01_pg_stat_statements_before.txt), [after](sql-tuning/05_pg_stat_statements_after.txt)).

**Measurement caveats.**
- **Machine.** These were measured on a laptop running Docker Desktop. The `clinical_documents` count query,
  which nothing in this change touches, read 0.61 ms in the first run and 3.30 ms in the second, so treat
  differences of a few milliseconds as noise.
- **What the evidence rests on.** The plan shapes and buffer counts carry the result. The 30-1500× speed-ups are
  far outside that noise.
- **What's left in Q2.** Its remaining 714 buffers are the 225 primary-key probes into `patients`, which the
  response needs.

## 6. Rolling it out to a live database

- **Build the indexes without blocking writers.** On a big, busy table use
  `CREATE INDEX CONCURRENTLY ... WHERE ...` outside a transaction. The partial indexes here built in 144 ms and
  186 ms on 300k rows because they are tiny, which is why the migration uses a plain `op.create_index`.
- **Refresh statistics.** Run `ANALYZE visits` right after the build, then re-check the plans.
- **Check the indexes in production.** `pg_stat_statements` is available on managed Postgres such as Neon's free
  tier. Run the same top-N query after deploy and confirm with `pg_stat_user_indexes.idx_scan` that the new
  indexes are actually used.

## 7. Reproduce

The whole setup runs on the free tier: Docker plus the local Python environment.

```bash
docker run -d --name triage-pg-tuning --shm-size=1g -e POSTGRES_PASSWORD=tuning -e POSTGRES_DB=triage -p 5433:5432 \
  postgres:16 -c shared_preload_libraries=pg_stat_statements -c track_io_timing=on
docker exec triage-pg-tuning psql -U postgres -d triage -c "CREATE EXTENSION pg_stat_statements"
export DATABASE_URL=postgresql://postgres:tuning@localhost:5433/triage
python -c "from backend.app.db_migrate import run_migrations; run_migrations()"   # applies 0001..0004
python -c "from backend.app.db_migrate import alembic_config; from alembic import command; command.downgrade(alembic_config(), '0003')"  # 'before' state
docker exec -i triage-pg-tuning psql -U postgres -d triage < backend/scripts/sql_tuning/generate_load.sql
python -m backend.scripts.sql_tuning.workload --rounds 300        # then the pg_stat_statements query above
docker exec -i triage-pg-tuning psql -U postgres -d triage < backend/scripts/sql_tuning/explain_queries.sql
```

**`--shm-size=1g` matters.** Docker's default 64 MB `/dev/shm` made `VACUUM` fail with
`could not resize shared memory segment ... No space left on device` once parallel workers kicked in. See the
[runbook, incident 4](runbook.md#incident-4-waiting-room-slow-under-production-sized-history).

## 8. Next candidates (not done)

- **Archive query.** `GET /api/archive` filters `discharge_time >= now() - 7 days` per facility. There is no
  index on `discharge_time` yet. Trace it under an archive-heavy workload before adding
  `(facility_id, discharge_time)`.
- **Retention aggregate.** The retention job groups all visits by `patient_id` once per run. That is a single
  sequential scan per run, acceptable for a nightly job, so it was left alone.
