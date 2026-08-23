CREATE INDEX IF NOT EXISTS idx_chunks_text_fts
    ON chunks USING gin (to_tsvector('simple', text));

CREATE INDEX IF NOT EXISTS idx_documents_name_fts
    ON documents USING gin (to_tsvector('simple', document_name));

