"""pgvector-backed policy store with hybrid (vector + full-text) retrieval.

Access control is enforced inside the SQL: a chunk is only a candidate if its
`allowed_roles` overlaps the caller's effective roles, so restricted text never leaves
the database for an unauthorized user.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import asyncpg
import numpy as np

from assistant.rag.chunker import Chunk, SourceDocument


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_id: str
    title: str
    heading: str
    text: str
    similarity: float
    score: float
    version: str | None = None


class VectorStore(Protocol):
    async def search(
        self, embedding: Sequence[float], query_text: str, roles: Sequence[str], k: int
    ) -> list[RetrievedChunk]: ...


# Reciprocal rank fusion of the vector and keyword candidate lists (k=60 is the usual constant).
_SEARCH_SQL = """
WITH vec AS (
    SELECT id, similarity, row_number() OVER (ORDER BY similarity DESC) AS rnk
    FROM (
        SELECT c.id, 1 - (c.embedding <=> $1) AS similarity
        FROM chunks c
        WHERE c.allowed_roles && $2::text[]
        ORDER BY c.embedding <=> $1
        LIMIT $4
    ) v
),
kw AS (
    SELECT id, row_number() OVER (ORDER BY rank DESC) AS rnk
    FROM (
        SELECT c.id, ts_rank_cd(c.tsv, q) AS rank
        FROM chunks c, websearch_to_tsquery('english', $3) q
        WHERE c.allowed_roles && $2::text[] AND c.tsv @@ q
        ORDER BY rank DESC
        LIMIT $4
    ) k
)
SELECT c.id, c.doc_id, d.title, c.heading, c.content, d.version,
       1 - (c.embedding <=> $1) AS similarity,
       COALESCE(1.0 / (60 + vec.rnk), 0) + COALESCE(1.0 / (60 + kw.rnk), 0) AS score
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
LEFT JOIN vec ON vec.id = c.id
LEFT JOIN kw ON kw.id = c.id
WHERE (vec.id IS NOT NULL AND vec.similarity >= $6) OR kw.id IS NOT NULL
ORDER BY score DESC
LIMIT $5
"""


class PgVectorStore:
    def __init__(self, pool: asyncpg.Pool, min_similarity: float = 0.25, candidates: int = 30):
        self._pool = pool
        self._min_similarity = min_similarity
        self._candidates = candidates

    async def search(
        self, embedding: Sequence[float], query_text: str, roles: Sequence[str], k: int
    ) -> list[RetrievedChunk]:
        if not roles:
            return []
        vec = np.asarray(embedding, dtype=np.float32)
        async with self._pool.acquire() as conn, conn.transaction():
            # Keep scanning the HNSW index until enough rows pass the role filter (pgvector >= 0.8).
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = await conn.fetch(
                _SEARCH_SQL, vec, list(roles), query_text, self._candidates, k, self._min_similarity
            )
        return [
            RetrievedChunk(
                chunk_id=r["id"],
                doc_id=r["doc_id"],
                title=r["title"],
                heading=r["heading"],
                text=r["content"],
                similarity=float(r["similarity"]),
                score=float(r["score"]),
                version=r["version"],
            )
            for r in rows
        ]

    async def document_hash(self, doc_id: str) -> str | None:
        return await self._pool.fetchval("SELECT content_hash FROM documents WHERE doc_id = $1", doc_id)

    async def upsert_document(
        self, doc: SourceDocument, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO documents (doc_id, title, source, version, owner, classification, allowed_roles,
                                       content_hash, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, now())
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title, source = EXCLUDED.source, version = EXCLUDED.version,
                    owner = EXCLUDED.owner, classification = EXCLUDED.classification,
                    allowed_roles = EXCLUDED.allowed_roles, content_hash = EXCLUDED.content_hash,
                    updated_at = now()
                """,
                doc.doc_id,
                doc.title,
                doc.source,
                doc.version,
                doc.owner,
                doc.classification,
                doc.allowed_roles,
                doc.content_hash,
            )
            await conn.execute("DELETE FROM chunks WHERE doc_id = $1", doc.doc_id)
            await conn.executemany(
                """
                INSERT INTO chunks (doc_id, chunk_index, heading, content, allowed_roles, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                [
                    (doc.doc_id, c.index, c.heading, c.text, doc.allowed_roles, np.asarray(e, dtype=np.float32))
                    for c, e in zip(chunks, embeddings, strict=True)
                ],
            )

    async def delete_missing(self, keep_doc_ids: set[str]) -> list[str]:
        rows = await self._pool.fetch(
            "DELETE FROM documents WHERE NOT (doc_id = ANY($1::text[])) RETURNING doc_id", list(keep_doc_ids)
        )
        return [r["doc_id"] for r in rows]
