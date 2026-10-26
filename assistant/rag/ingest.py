"""Ingest policy documents into the vector store.

    python -m assistant.rag.ingest data/policies          # add / update changed documents
    python -m assistant.rag.ingest data/policies --prune  # also remove documents no longer on disk

Unchanged documents (same content hash) are skipped. Any change bumps the knowledge-base
version, which invalidates semantic-cache entries built on the old content.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import redis.asyncio as redis

from assistant.cache.semantic import bump_kb_version
from assistant.config import get_settings
from assistant.db import create_pool, migrate
from assistant.rag.chunker import chunk_document, load_document
from assistant.rag.embeddings import Embedder, build_embedder
from assistant.rag.store import PgVectorStore

log = logging.getLogger("assistant.ingest")
SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


async def ingest(paths: list[Path], store: PgVectorStore, embedder: Embedder, prune: bool = False) -> dict:
    files = sorted(f for p in paths for f in ([p] if p.is_file() else p.rglob("*")) if f.suffix.lower() in SUFFIXES)
    stats = {"updated": [], "unchanged": [], "removed": []}
    seen: set[str] = set()
    for path in files:
        doc = load_document(path)
        if doc.doc_id in seen:
            raise ValueError(f"duplicate doc_id {doc.doc_id!r} ({path})")
        seen.add(doc.doc_id)
        if await store.document_hash(doc.doc_id) == doc.content_hash:
            stats["unchanged"].append(doc.doc_id)
            continue
        chunks = chunk_document(doc)
        embeddings = await embedder.embed([c.embedding_text(doc.title) for c in chunks])
        await store.upsert_document(doc, chunks, embeddings)
        stats["updated"].append(doc.doc_id)
        log.info("ingested %s (%d chunks, roles=%s)", doc.doc_id, len(chunks), doc.allowed_roles)
    if prune:
        stats["removed"] = await store.delete_missing(seen)
    return stats


async def _main(paths: list[Path], prune: bool) -> None:
    settings = get_settings()
    pool = await create_pool(settings.database_url)
    r = redis.from_url(settings.redis_url)
    try:
        await migrate(pool, settings.embedding_dim)
        store = PgVectorStore(pool, settings.rag_min_similarity)
        stats = await ingest(paths, store, build_embedder(settings), prune=prune)
        if stats["updated"] or stats["removed"]:
            version = await bump_kb_version(r)
            log.info("knowledge base version is now %s", version)
        log.info(
            "updated=%d unchanged=%d removed=%d",
            len(stats["updated"]),
            len(stats["unchanged"]),
            len(stats["removed"]),
        )
    finally:
        await pool.close()
        await r.aclose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--prune", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main(args.paths, args.prune))


if __name__ == "__main__":
    main()
