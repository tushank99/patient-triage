-- Synthetic, production-sized EHR history for SQL tuning (Postgres only, LOCAL ONLY).
-- Shape: 50k patients, ~1M historical clinical documents (avg 20 per patient, skewed),
-- 300k historical ED visits of which a few hundred are still in the live queue.
-- Documents are inserted in date order (as a real EHR accumulates them), so one patient's
-- history is scattered across the heap, like in production.
-- Run against an Alembic-migrated, EMPTY database:
--   docker exec -i triage-pg-tuning psql -U postgres -d triage < backend/scripts/sql_tuning/generate_load.sql

\timing on
SET synchronous_commit = off;
SELECT setseed(0.42);

INSERT INTO patients (id, mrn, name, date_of_birth, age_years, gender, history_facts, is_anonymized, created_at)
SELECT 'pat-' || g,
       'MRN-' || lpad(g::text, 7, '0'),
       'Synthetic Patient ' || g,
       date '1940-01-01' + (random() * 30000)::int,
       round((random() * 85)::numeric, 2),
       (ARRAY['M','F','O'])[1 + (g % 3)],
       NULL, false,
       now() - (random() * interval '3650 days')
FROM generate_series(1, 50000) g;

-- Skewed: a few chronic patients have hundreds of documents, most have a handful.
INSERT INTO clinical_documents (id, patient_id, doc_type, title, doc_date, content, checksum, source, is_synthetic,
                                status, current_version, filename, mime_type, uploaded_by, created_at, updated_at)
SELECT 'doc-' || d.n,
       'pat-' || (1 + floor(power(random(), 2.2) * 50000))::int,
       (ARRAY['clinic_note','discharge_summary','lab_result','imaging_report','medication_list','problem_list'])[1 + (d.n % 6)],
       'Historical document ' || d.n,
       d.doc_date,
       repeat('Synthetic clinical narrative. ', 12 + (d.n % 8)),
       md5(d.n::text),
       'synthetic-load',
       true,
       CASE WHEN r < 0.90 THEN 'READY' WHEN r < 0.95 THEN 'DELETED' WHEN r < 0.98 THEN 'FAILED' ELSE 'NEEDS_REVIEW' END,
       1, NULL, 'text/plain', 'loader',
       d.doc_date::timestamptz, d.doc_date::timestamptz
FROM (
    SELECT n, random() AS r, date '2016-01-01' + (n::bigint * 3650 / 1000000)::int AS doc_date
    FROM generate_series(1, 1000000) n
) d;

INSERT INTO visits (id, patient_id, facility_id, arrival_time, chief_complaint, heart_rate, systolic_bp, diastolic_bp,
                    respiratory_rate, oxygen_saturation, temperature_celsius, pain_score, ai_esi_score, final_esi_score,
                    status, needs_review, is_overridden, is_demo_surge, discharge_time)
SELECT 'visit-' || v,
       'pat-' || (1 + floor(random() * 50000))::int,
       CASE WHEN v % 4 = 0 THEN 'ED-NORTH' ELSE 'ED-MAIN' END,
       ts, 'Synthetic complaint ' || (v % 50),
       60 + (random() * 60)::int, 90 + (random() * 60)::int, 60 + (random() * 30)::int,
       12 + (random() * 12)::int, 90 + random() * 10, 36 + random() * 3, (random() * 10)::int,
       1 + (random() * 4)::int, 1 + (random() * 4)::int,
       CASE WHEN v > 299700 THEN (ARRAY['WAITING','NEEDS_REASSESSMENT'])[1 + (v % 2)] ELSE 'DISCHARGED' END,
       false, false, false,
       CASE WHEN v > 299700 THEN NULL ELSE ts + interval '4 hours' END
FROM (SELECT v, timestamptz '2016-01-01' + (v * interval '17 minutes') AS ts FROM generate_series(1, 300000) v) s;

VACUUM ANALYZE patients;
VACUUM ANALYZE clinical_documents;
VACUUM ANALYZE visits;

SELECT relname, n_live_tup, pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_stat_user_tables WHERE relname IN ('patients','clinical_documents','visits') ORDER BY relname;
