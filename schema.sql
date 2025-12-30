BEGIN;

CREATE SCHEMA IF NOT EXISTS inference;

CREATE TABLE inference_jobs (
    job_id UUID PRIMARY KEY,
    rut_comercio TEXT NOT NULL,
    inference_type TEXT NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,        -- PENDING | RUNNING | DONE | FAILED
    created_at TIMESTAMP DEFAULT now(),
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    result JSONB,
    error TEXT
);

COMMIT;