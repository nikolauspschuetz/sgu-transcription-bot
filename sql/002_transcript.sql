-- Transcript timestream — the episodes as a time-ordered, searchable store (#13).
--
-- Each row is one diarized segment (start/end/speaker/text). Ordered by
-- (episode_number, start_s) it reads as a time-series ("timestream"); the generated
-- tsvector + GIN index makes the whole corpus full-text searchable. A semantic-search
-- embedding column (pgvector) can be added later for RAG (the SGU Bot).

CREATE TABLE IF NOT EXISTS transcript_segments (
    episode_number int              NOT NULL,
    seq            int              NOT NULL,
    start_s        double precision NOT NULL,
    end_s          double precision NOT NULL,
    speaker        text             NOT NULL,
    text           text             NOT NULL,
    tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    PRIMARY KEY (episode_number, seq)
);

CREATE INDEX IF NOT EXISTS transcript_segments_tsv_idx  ON transcript_segments USING gin (tsv);
CREATE INDEX IF NOT EXISTS transcript_segments_time_idx ON transcript_segments (episode_number, start_s);
CREATE INDEX IF NOT EXISTS transcript_segments_spk_idx  ON transcript_segments (speaker);
