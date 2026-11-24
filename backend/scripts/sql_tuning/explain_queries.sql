-- The three slowest statements from pg_stat_statements (see docs/sql-tuning.md), with the parameters bound.
-- The app selects every visits column; visits.* gives the same plan. Each query runs once to warm the
-- cache, then EXPLAIN (ANALYZE, BUFFERS) measures a warm run, so before/after compare plans, not disk luck.
\pset pager off
\set pid '''pat-12345'''

\echo '=== Q1 retention: active visits for one patient (ops/retention.py) ==='
SELECT count(*) FROM (SELECT visits.* FROM visits WHERE visits.patient_id = :pid AND visits.status IN ('WAITING', 'NEEDS_REASSESSMENT')) anon_1;
EXPLAIN (ANALYZE, BUFFERS)
SELECT count(*) FROM (SELECT visits.* FROM visits WHERE visits.patient_id = :pid AND visits.status IN ('WAITING', 'NEEDS_REASSESSMENT')) anon_1;

\echo '=== Q2 waiting room for a facility (routers/queue.py) ==='
SELECT count(*) FROM (SELECT visits.* FROM visits JOIN patients ON patients.id = visits.patient_id
 WHERE visits.status IN ('WAITING', 'NEEDS_REASSESSMENT') AND visits.facility_id = 'ED-MAIN'
 ORDER BY visits.final_esi_score ASC, visits.arrival_time ASC) w;
EXPLAIN (ANALYZE, BUFFERS)
SELECT visits.* FROM visits JOIN patients ON patients.id = visits.patient_id
 WHERE visits.status IN ('WAITING', 'NEEDS_REASSESSMENT') AND visits.facility_id = 'ED-MAIN'
 ORDER BY visits.final_esi_score ASC, visits.arrival_time ASC;

\echo '=== Q3 deterioration monitor tick (background_tasks.py) ==='
SELECT count(*) FROM visits WHERE visits.status = 'WAITING';
EXPLAIN (ANALYZE, BUFFERS)
SELECT visits.* FROM visits WHERE visits.status = 'WAITING';
