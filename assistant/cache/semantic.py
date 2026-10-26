"""Semantic answer cache on Redis vector search.

An entry is only ever served inside its *scope*: a hash of the caller's effective roles,
the knowledge-base version and the system-prompt version. Two users with different
document access can therefore never share an answer, and re-ingesting policies or editing
the persona retires old answers automatically.

Only grounded policy answers are cached; anything that used tools, contained personal
data, or was blocked is not (see `Assistant._cacheable`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

log = logging.getLogger(__name__)

KB_VERSION_KEY = "kb:version"


@dataclass
class CachedAnswer:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    query: str = ""
    similarity: float = 1.0


class SemanticCache(Protocol):
    async def scope(self, roles: frozenset[str], prompt_version: str) -> str: ...

    async def lookup(self, scope: str, embedding: list[float]) -> CachedAnswer | None: ...

    async def store(self, scope: str, embedding: list[float], entry: CachedAnswer) -> None: ...


async def bump_kb_version(r: redis.Redis) -> int:
    return int(await r.incr(KB_VERSION_KEY))


def scope_key(roles: frozenset[str], kb_version: str, prompt_version: str) -> str:
    raw = f"{kb_version}|{prompt_version}|{'|'.join(sorted(roles))}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class RedisSemanticCache:
    def __init__(
        self,
        client: redis.Redis,
        dim: int,
        threshold: float,
        ttl_seconds: int,
        index: str = "idx:semcache",
        prefix: str = "semcache:",
    ):
        self._r = client
        self._dim = dim
        self._threshold = threshold
        self._ttl = ttl_seconds
        self.index = index
        self.prefix = prefix

    async def ensure_index(self) -> None:
        try:
            await self._r.ft(self.index).info()
        except ResponseError:
            await self._r.ft(self.index).create_index(
                fields=[
                    TagField("scope"),
                    VectorField(
                        "embedding",
                        "HNSW",
                        {"TYPE": "FLOAT32", "DIM": self._dim, "DISTANCE_METRIC": "COSINE"},
                    ),
                ],
                definition=IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH),
            )

    async def scope(self, roles: frozenset[str], prompt_version: str) -> str:
        raw = await self._r.get(KB_VERSION_KEY)
        kb_version = raw.decode() if isinstance(raw, bytes) else (raw or "0")
        return scope_key(roles, kb_version, prompt_version)

    async def lookup(self, scope: str, embedding: list[float]) -> CachedAnswer | None:
        query = (
            Query(f"(@scope:{{{scope}}})=>[KNN 1 @embedding $vec AS distance]")
            .sort_by("distance")
            .return_fields("entry", "distance")
            .dialect(2)
        )
        res = await self._r.ft(self.index).search(
            query, query_params={"vec": np.asarray(embedding, dtype=np.float32).tobytes()}
        )
        if not res.docs:
            return None
        doc = res.docs[0]
        similarity = 1.0 - float(doc.distance)
        if similarity < self._threshold:
            return None
        raw = doc.entry.decode() if isinstance(doc.entry, bytes) else doc.entry
        entry = CachedAnswer(**json.loads(raw))
        entry.similarity = similarity
        return entry

    async def store(self, scope: str, embedding: list[float], entry: CachedAnswer) -> None:
        key = f"{self.prefix}{uuid.uuid4().hex}"
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.hset(
                key,
                mapping={
                    "scope": scope,
                    "embedding": np.asarray(embedding, dtype=np.float32).tobytes(),
                    "entry": json.dumps(asdict(entry)),
                },
            )
            pipe.expire(key, self._ttl)
            await pipe.execute()
