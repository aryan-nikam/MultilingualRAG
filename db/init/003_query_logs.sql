CREATE TABLE IF NOT EXISTS query_logs (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    query_language TEXT,
    retrieval_mode TEXT NOT NULL,
    rerank BOOLEAN NOT NULL,
    retrieved_chunks JSONB NOT NULL,
    answer TEXT,
    citations JSONB NOT NULL,
    validation JSONB NOT NULL,
    latency_ms DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_query_logs_language ON query_logs(query_language);
