"""Against real Postgres + pgvector and Redis (docker compose up -d postgres redis).

    pytest -m integration

Uses a throwaway database and a uniquely named Redis index, both removed afterwards.
"""

import os
import uuid

import asyncpg
import pytest
import redis.asyncio as redis

from assistant.cache.semantic import CachedAnswer, RedisSemanticCache, bump_kb_version, scope_key
from assistant.db import create_pool, migrate
from assistant.llm import Usage
from assistant.observability.prompt_log import PostgresPromptLogger, PromptLogRecord
from assistant.rag.embeddings import HashingEmbedder
from assistant.rag.ingest import ingest
from assistant.rag.store import PgVectorStore
from assistant.sessions import PendingAction, PgRedisSessionStore
from tests.fakes import ROOT

pytestmark = pytest.mark.integration

ADMIN_DSN = os.environ.get("DATABASE_URL", "postgresql://assistant:assistant@localhost:5432/assistant")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DIM = 256


@pytest.fixture
async def pool():
    name = f"assistant_it_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(ADMIN_DSN)
    await admin.execute(f'CREATE DATABASE "{name}"')
    dsn = ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}"
    p = await create_pool(dsn)
    await migrate(p, DIM)
    try:
        yield p
    finally:
        await p.close()
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()


@pytest.fixture
async def r():
    client = redis.from_url(REDIS_URL)
    yield client
    await client.aclose()


async def test_ingest_and_role_filtered_hybrid_search(pool):
    store = PgVectorStore(pool, min_similarity=0.05)
    embedder = HashingEmbedder(DIM)
    stats = await ingest([ROOT / "data/policies"], store, embedder)
    assert "hr-comp-001" in stats["updated"]
    again = await ingest([ROOT / "data/policies"], store, embedder)
    assert again["updated"] == [] and len(again["unchanged"]) == len(stats["updated"])

    q = "engineering salary bands level E4"
    [vec] = await embedder.embed([q])
    as_employee = await store.search(vec, q, ["employee"], k=5)
    as_hr = await store.search(vec, q, ["employee", "hr"], k=5)
    assert "hr-comp-001" not in {c.doc_id for c in as_employee}
    assert as_hr[0].doc_id == "hr-comp-001"

    q = "carry over unused leave 31 March"
    [vec] = await embedder.embed([q])
    top = await store.search(vec, q, ["employee"], k=3)
    assert top[0].doc_id == "hr-leave-001" and "Carry-over" in top[0].heading


async def test_migrate_refuses_dimension_change(pool):
    with pytest.raises(RuntimeError, match="dimension"):
        await migrate(pool, DIM + 1)


async def test_semantic_cache_scoping_and_threshold(r):
    tag = uuid.uuid4().hex[:8]
    cache = RedisSemanticCache(r, DIM, threshold=0.9, ttl_seconds=60, index=f"idx:semcache-{tag}", prefix=f"sc-{tag}:")
    await cache.ensure_index()
    try:
        embedder = HashingEmbedder(DIM)
        [v1, v2] = await embedder.embed(["how many leave days carry over", "hotel limit in london"])
        employee = scope_key(frozenset({"employee"}), "1", "p")
        hr = scope_key(frozenset({"employee", "hr"}), "1", "p")
        await cache.store(employee, v1, CachedAnswer("5 days [1]", [{"doc_id": "hr-leave-001"}], "m", "q"))

        hit = await cache.lookup(employee, v1)
        assert hit is not None and hit.answer == "5 days [1]" and hit.similarity > 0.99
        assert await cache.lookup(hr, v1) is None  # other role set
        assert await cache.lookup(employee, v2) is None  # not similar enough
        assert await r.ttl(next(iter(await r.keys(f"sc-{tag}:*")))) > 0
    finally:
        await r.ft(cache.index).dropindex(delete_documents=True)


async def test_kb_version_changes_cache_scope(r):
    roles = frozenset({"employee"})
    cache = RedisSemanticCache(r, DIM, 0.9, 60)
    before = await cache.scope(roles, "p")
    await bump_kb_version(r)
    assert await cache.scope(roles, "p") != before


async def test_sessions_vault_and_single_use_actions(pool, r):
    sessions = PgRedisSessionStore(pool, r, vault_ttl_seconds=60, action_ttl_seconds=60)
    conv = await sessions.get_or_create(None, "dev-alice")
    conv.vault.redact("mail bob@acme.example")
    await sessions.append(conv, "user", "mail <EMAIL_1>")
    await sessions.save_vault(conv)

    loaded = await sessions.get(conv.id, "dev-alice")
    assert loaded.history == [("user", "mail <EMAIL_1>")]
    assert loaded.vault.restore("<EMAIL_1>") == "bob@acme.example"
    assert await sessions.get(conv.id, "dev-mallory") is None
    stored = await pool.fetchval("SELECT content FROM conversation_messages WHERE conversation_id = $1", uuid.UUID(conv.id))
    assert "bob@" not in stored

    action = PendingAction(uuid.uuid4().hex, "dev-alice", conv.id, "hr__submit_leave_request", {"a": 1})
    await sessions.put_action(action)
    assert await sessions.pop_action("dev-mallory", action.id) is None
    assert (await sessions.pop_action("dev-alice", action.id)).arguments == {"a": 1}
    assert await sessions.pop_action("dev-alice", action.id) is None


async def test_prompt_log_round_trip(pool):
    logger = PostgresPromptLogger(pool)
    logger.write(
        PromptLogRecord(
            kind="chat",
            user_sub="dev-alice",
            roles=["employee"],
            input_redacted="hi <EMAIL_1>",
            tool_calls=[{"name": "hr__get_leave_balance", "status": "ok"}],
            usage=Usage(10, 5),
            cost_usd=0.0001,
        )
    )
    await logger.drain()
    [row] = await logger.recent(5)
    assert row["input_redacted"] == "hi <EMAIL_1>" and row["tool_calls"][0]["status"] == "ok"
