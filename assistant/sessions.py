"""Conversation state.

- History (redacted text only) lives in Postgres.
- The PII vault (placeholder -> real value) lives in Redis with a TTL, so raw personal data
  is never written to the database and expires with the conversation.
- Pending side-effecting actions live in Redis, keyed by user, until confirmed or expired.

Earlier turns are replayed to the model as plain text. Provider-native content (such as
Claude thinking blocks) is only replayed within a single turn's tool loop, so history is
never edited after the fact.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import asyncpg
import redis.asyncio as redis

from assistant.guardrails.pii import PIIVault
from assistant.llm import Message


@dataclass
class Conversation:
    id: str
    user_sub: str
    history: list[tuple[str, str]] = field(default_factory=list)  # (role, redacted text)
    vault: PIIVault = field(default_factory=PIIVault)

    def history_messages(self, max_turns: int) -> list[Message]:
        msgs = self.history[-2 * max_turns :]
        while msgs and msgs[0][0] != "user":  # a transcript must start with a user turn
            msgs = msgs[1:]
        return [Message(role=role, content=text) for role, text in msgs]  # type: ignore[arg-type]


@dataclass
class PendingAction:
    id: str
    user_sub: str
    conversation_id: str
    tool_name: str
    arguments: dict[str, Any]  # redacted; placeholders are restored from the vault at execution
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())


class SessionStore(Protocol):
    async def get_or_create(self, conversation_id: str | None, user_sub: str) -> Conversation: ...

    async def get(self, conversation_id: str, user_sub: str) -> Conversation | None: ...

    async def append(self, conversation: Conversation, role: str, content: str) -> None: ...

    async def save_vault(self, conversation: Conversation) -> None: ...

    async def put_action(self, action: PendingAction) -> None: ...

    async def pop_action(self, user_sub: str, action_id: str) -> PendingAction | None: ...

    async def allow_request(self, user_sub: str, limit_per_minute: int) -> bool: ...


class ConversationNotFound(Exception):
    pass


class PgRedisSessionStore:
    def __init__(self, pool: asyncpg.Pool, client: redis.Redis, vault_ttl_seconds: int, action_ttl_seconds: int):
        self._pool = pool
        self._r = client
        self._vault_ttl = vault_ttl_seconds
        self._action_ttl = action_ttl_seconds

    async def get_or_create(self, conversation_id: str | None, user_sub: str) -> Conversation:
        if conversation_id:
            conv = await self.get(conversation_id, user_sub)
            if conv is None:
                raise ConversationNotFound(conversation_id)
            return conv
        conv = Conversation(id=str(uuid.uuid4()), user_sub=user_sub)
        await self._pool.execute(
            "INSERT INTO conversations (id, user_sub) VALUES ($1, $2)", uuid.UUID(conv.id), user_sub
        )
        return conv

    async def get(self, conversation_id: str, user_sub: str) -> Conversation | None:
        try:
            cid = uuid.UUID(conversation_id)
        except ValueError:
            return None
        owner = await self._pool.fetchval("SELECT user_sub FROM conversations WHERE id = $1", cid)
        if owner != user_sub:  # someone else's conversation looks exactly like a missing one
            return None
        rows = await self._pool.fetch(
            "SELECT role, content FROM conversation_messages WHERE conversation_id = $1 ORDER BY id", cid
        )
        raw = await self._r.get(f"vault:{conversation_id}")
        vault = PIIVault(json.loads(raw) if raw else None)
        return Conversation(conversation_id, user_sub, [(r["role"], r["content"]) for r in rows], vault)

    async def append(self, conversation: Conversation, role: str, content: str) -> None:
        conversation.history.append((role, content))
        cid = uuid.UUID(conversation.id)
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO conversation_messages (conversation_id, role, content) VALUES ($1, $2, $3)",
                cid,
                role,
                content,
            )
            await conn.execute("UPDATE conversations SET updated_at = now() WHERE id = $1", cid)

    async def save_vault(self, conversation: Conversation) -> None:
        if len(conversation.vault):
            await self._r.set(
                f"vault:{conversation.id}", json.dumps(conversation.vault.to_dict()), ex=self._vault_ttl
            )

    async def put_action(self, action: PendingAction) -> None:
        await self._r.set(f"action:{action.user_sub}:{action.id}", json.dumps(asdict(action)), ex=self._action_ttl)

    async def pop_action(self, user_sub: str, action_id: str) -> PendingAction | None:
        # Keyed by user and fetched with GETDEL: an action can be confirmed once, only by its owner.
        raw = await self._r.getdel(f"action:{user_sub}:{action_id}")
        return PendingAction(**json.loads(raw)) if raw else None

    async def allow_request(self, user_sub: str, limit_per_minute: int) -> bool:
        window = int(dt.datetime.now(dt.timezone.utc).timestamp() // 60)
        key = f"ratelimit:{user_sub}:{window}"
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, 120)
            count, _ = await pipe.execute()
        return int(count) <= limit_per_minute
