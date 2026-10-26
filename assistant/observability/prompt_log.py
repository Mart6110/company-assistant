"""Prompt log: one row per turn or confirmed action, with redacted input/output, retrieval,
tool calls, guardrail decisions, tokens, cost and latency. Joined to traces by trace_id.

Writes happen off the request path; a failed write is logged, never surfaced to the user.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import asyncpg
import yaml

from assistant.llm import Usage

log = logging.getLogger(__name__)


@dataclass
class PromptLogRecord:
    kind: str  # "chat" | "action"
    user_sub: str
    roles: list[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str | None = None
    conversation_id: str | None = None
    route_tier: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    input_redacted: str | None = None
    output_redacted: str | None = None
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    guardrail_events: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: int = 0
    cache_hit: bool = False
    blocked: bool = False
    block_reason: str | None = None
    error: str | None = None


class PromptLogger(Protocol):
    def write(self, record: PromptLogRecord) -> None: ...

    async def drain(self) -> None: ...


class PostgresPromptLogger:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._pending: set[asyncio.Task] = set()

    def write(self, record: PromptLogRecord) -> None:
        task = asyncio.create_task(self._insert(record))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    async def _insert(self, r: PromptLogRecord) -> None:
        try:
            await self._pool.execute(
                """
                INSERT INTO prompt_logs (id, trace_id, kind, conversation_id, user_sub, roles, route_tier, model,
                    prompt_version, input_redacted, output_redacted, retrieved, tool_calls, guardrail_events,
                    input_tokens, output_tokens, cache_read_tokens, cost_usd, latency_ms, cache_hit, blocked,
                    block_reason, error)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13::jsonb, $14::jsonb,
                    $15, $16, $17, $18, $19, $20, $21, $22, $23)
                """,
                uuid.UUID(r.id),
                r.trace_id,
                r.kind,
                uuid.UUID(r.conversation_id) if r.conversation_id else None,
                r.user_sub,
                r.roles,
                r.route_tier,
                r.model,
                r.prompt_version,
                r.input_redacted,
                r.output_redacted,
                json.dumps(r.retrieved),
                json.dumps(r.tool_calls, default=str),
                json.dumps(r.guardrail_events),
                r.usage.input_tokens,
                r.usage.output_tokens,
                r.usage.cache_read_tokens,
                r.cost_usd,
                r.latency_ms,
                r.cache_hit,
                r.blocked,
                r.block_reason,
                r.error,
            )
        except Exception:
            log.exception("failed to write prompt log %s", r.id)

    async def recent(self, limit: int = 50, user_sub: str | None = None) -> list[dict[str, Any]]:
        rows = await self._pool.fetch(
            """
            SELECT * FROM prompt_logs
            WHERE ($2::text IS NULL OR user_sub = $2)
            ORDER BY created_at DESC LIMIT $1
            """,
            limit,
            user_sub,
        )
        out = []
        for row in rows:
            item = dict(row)
            for key in ("retrieved", "tool_calls", "guardrail_events"):
                item[key] = json.loads(item[key]) if isinstance(item[key], str) else item[key]
            item["id"] = str(item["id"])
            item["conversation_id"] = str(item["conversation_id"]) if item["conversation_id"] else None
            item["cost_usd"] = float(item["cost_usd"])
            item["created_at"] = item["created_at"].isoformat()
            out.append(item)
        return out


class Pricing:
    """USD per million tokens, keyed by "<provider>:<model>" (config/pricing.yaml)."""

    def __init__(self, table: dict[str, dict[str, float]]):
        self._table = table

    @classmethod
    def load(cls, path: str | Path) -> Pricing:
        p = Path(path)
        if not p.exists():
            return cls({})
        return cls(yaml.safe_load(p.read_text()) or {})

    def cost(self, model: str, usage: Usage) -> float:
        price = self._table.get(model)
        if not price:
            return 0.0
        # Anthropic reports cache reads separately from input_tokens; OpenAI includes them.
        if model.startswith("anthropic:"):
            uncached = usage.input_tokens
        else:
            uncached = max(usage.input_tokens - usage.cache_read_tokens, 0)
        total = (
            uncached * price.get("input", 0)
            + usage.cache_read_tokens * price.get("cache_read", price.get("input", 0) * 0.1)
            + usage.cache_write_tokens * price.get("cache_write", price.get("input", 0) * 1.25)
            + usage.output_tokens * price.get("output", 0)
        )
        return round(total / 1_000_000, 6)
