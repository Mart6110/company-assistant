"""Composition root: builds the Assistant and its infrastructure from Settings.
Used by the API and the eval runner, so evals exercise exactly what production runs."""

from __future__ import annotations

from dataclasses import dataclass

import asyncpg
import redis.asyncio as redis

from assistant.auth.rbac import RBACPolicy
from assistant.auth.tokens import TokenValidator, build_validator
from assistant.cache.semantic import RedisSemanticCache
from assistant.config import Settings
from assistant.db import create_pool, migrate
from assistant.guardrails import Guardrails, InputClassifier
from assistant.llm import ModelRef, ProviderRegistry
from assistant.observability.prompt_log import PostgresPromptLogger, Pricing
from assistant.orchestrator import Assistant
from assistant.prompts import Persona
from assistant.rag.embeddings import build_embedder
from assistant.rag.store import PgVectorStore
from assistant.router import ModelRouter
from assistant.sessions import PgRedisSessionStore
from assistant.tools.gateway import MCPToolGateway, TokenBroker, load_server_configs


@dataclass
class Runtime:
    assistant: Assistant
    validator: TokenValidator
    rbac: RBACPolicy
    pool: asyncpg.Pool
    redis: redis.Redis
    prompt_log: PostgresPromptLogger
    sessions: PgRedisSessionStore

    async def close(self) -> None:
        await self.prompt_log.drain()
        await self.pool.close()
        await self.redis.aclose()


async def build_runtime(settings: Settings) -> Runtime:
    persona = Persona.load(settings.persona_file)
    rbac = RBACPolicy.load(settings.rbac_file)
    llm = ProviderRegistry(settings)
    small = ModelRef.parse(settings.small_model, settings.small_effort)
    frontier = ModelRef.parse(settings.frontier_model, settings.frontier_effort)
    classifier = InputClassifier(llm, ModelRef.parse(settings.classifier_model_spec, settings.small_effort), persona.company)

    pool = await create_pool(settings.database_url)
    await migrate(pool, settings.embedding_dim)
    r = redis.from_url(settings.redis_url)  # binary-safe: vectors are stored as raw float32 bytes

    cache = None
    if settings.semantic_cache_enabled:
        cache = RedisSemanticCache(
            r, settings.embedding_dim, settings.semantic_cache_threshold, settings.semantic_cache_ttl_seconds
        )
        await cache.ensure_index()

    sessions = PgRedisSessionStore(
        pool, r, vault_ttl_seconds=86400, action_ttl_seconds=settings.pending_action_ttl_seconds
    )
    prompt_log = PostgresPromptLogger(pool)
    assistant = Assistant(
        settings=settings,
        persona=persona,
        rbac=rbac,
        llm=llm,
        router=ModelRouter(small, frontier, settings.router_threshold),
        guardrails=Guardrails(settings, classifier),
        embedder=build_embedder(settings),
        store=PgVectorStore(pool, settings.rag_min_similarity),
        tools=MCPToolGateway(load_server_configs(settings.mcp_servers_file), rbac, TokenBroker(settings)),
        sessions=sessions,
        prompt_log=prompt_log,
        pricing=Pricing.load(settings.pricing_file),
        cache=cache,
    )
    return Runtime(assistant, build_validator(settings), rbac, pool, r, prompt_log, sessions)
