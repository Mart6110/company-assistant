from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

# Idempotent schema. The embedding column's dimension is fixed at creation time; changing
# EMBEDDING_DIM later requires dropping `chunks` and re-ingesting.
_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          text PRIMARY KEY,
    title           text NOT NULL,
    source          text NOT NULL,
    version         text,
    owner           text,
    classification  text NOT NULL DEFAULT 'internal',
    allowed_roles   text[] NOT NULL,
    content_hash    text NOT NULL,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id              bigserial PRIMARY KEY,
    doc_id          text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index     int NOT NULL,
    heading         text NOT NULL,
    content         text NOT NULL,
    allowed_roles   text[] NOT NULL,
    embedding       vector({dim}) NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', heading || ' ' || content)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_roles_idx ON chunks USING gin (allowed_roles);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);

CREATE TABLE IF NOT EXISTS conversations (
    id          uuid PRIMARY KEY,
    user_sub    text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversations_user_idx ON conversations (user_sub, updated_at DESC);

-- Messages are stored redacted; the PII vault lives in Redis with a TTL.
CREATE TABLE IF NOT EXISTS conversation_messages (
    id               bigserial PRIMARY KEY,
    conversation_id  uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             text NOT NULL CHECK (role IN ('user', 'assistant')),
    content          text NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversation_messages_conv_idx ON conversation_messages (conversation_id, id);

CREATE TABLE IF NOT EXISTS prompt_logs (
    id                 uuid PRIMARY KEY,
    created_at         timestamptz NOT NULL DEFAULT now(),
    trace_id           text,
    kind               text NOT NULL,
    conversation_id    uuid,
    user_sub           text NOT NULL,
    roles              text[] NOT NULL,
    route_tier         text,
    model              text,
    prompt_version     text,
    input_redacted     text,
    output_redacted    text,
    retrieved          jsonb NOT NULL DEFAULT '[]',
    tool_calls         jsonb NOT NULL DEFAULT '[]',
    guardrail_events   jsonb NOT NULL DEFAULT '[]',
    input_tokens       int NOT NULL DEFAULT 0,
    output_tokens      int NOT NULL DEFAULT 0,
    cache_read_tokens  int NOT NULL DEFAULT 0,
    cost_usd           numeric(12, 6) NOT NULL DEFAULT 0,
    latency_ms         int NOT NULL DEFAULT 0,
    cache_hit          boolean NOT NULL DEFAULT false,
    blocked            boolean NOT NULL DEFAULT false,
    block_reason       text,
    error              text
);
CREATE INDEX IF NOT EXISTS prompt_logs_created_idx ON prompt_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS prompt_logs_user_idx ON prompt_logs (user_sub, created_at DESC);
"""


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def create_pool(dsn: str) -> asyncpg.Pool:
    # The extension must exist before the vector codec can be registered on pooled connections.
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await conn.close()
    return await asyncpg.create_pool(dsn, min_size=1, max_size=10, init=_init_connection)


async def migrate(pool: asyncpg.Pool, embedding_dim: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(_SCHEMA.format(dim=int(embedding_dim)))
        existing = await conn.fetchval(
            """
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
            """
        )
        if existing is not None and existing != embedding_dim:
            raise RuntimeError(
                f"chunks.embedding has dimension {existing} but EMBEDDING_DIM={embedding_dim}; "
                "drop the chunks table and re-run ingestion after changing embedding models"
            )
