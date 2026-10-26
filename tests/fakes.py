"""In-memory stand-ins for the infrastructure, so orchestration logic tests run without Docker."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from assistant.auth.rbac import Principal, RBACPolicy
from assistant.cache.semantic import CachedAnswer, scope_key
from assistant.config import Settings
from assistant.guardrails import Guardrails
from assistant.llm import LLMResponse, Message, ModelRef, ProviderRegistry, ToolCall, ToolSpec, Usage
from assistant.observability.prompt_log import Pricing, PromptLogRecord
from assistant.orchestrator import Assistant
from assistant.prompts import Persona
from assistant.rag.chunker import chunk_document, load_document
from assistant.rag.embeddings import HashingEmbedder
from assistant.rag.store import RetrievedChunk
from assistant.router import ModelRouter
from assistant.sessions import Conversation, ConversationNotFound, PendingAction
from assistant.tools import RegisteredTool

ROOT = Path(__file__).resolve().parent.parent


def settings(**overrides) -> Settings:
    base = dict(
        environment="test",
        auth_mode="dev",
        embedding_model="hashing",
        embedding_dim=256,
        tracing_enabled=False,
        persona_file=str(ROOT / "config/persona.yaml"),
        rbac_file=str(ROOT / "config/rbac.yaml"),
        guardrails_llm_classifier=False,
        semantic_cache_threshold=0.9,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def rbac() -> RBACPolicy:
    return RBACPolicy.load(ROOT / "config/rbac.yaml")


def principal(username: str = "alice", roles: tuple[str, ...] = ("employee",)) -> Principal:
    return rbac().principal(
        sub=f"dev-{username}",
        username=username,
        roles=roles,
        email=f"{username}@acme.example",
        given_name=username.title(),
        token="test-token",
    )


@dataclass
class ScriptedProvider:
    """Returns queued responses in order and records every request it receives."""

    name: str = "fake"
    responses: list[LLMResponse] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)

    async def complete(self, *, model, system, messages, tools=None, max_tokens=16000, effort=None, json_schema=None):
        self.requests.append(
            {"model": model, "system": system, "messages": [_copy(m) for m in messages], "tools": tools}
        )
        if not self.responses:
            raise AssertionError("ScriptedProvider ran out of responses")
        return self.responses.pop(0)


def _copy(m: Message) -> Message:
    return Message(m.role, m.content, list(m.tool_calls), list(m.tool_results), m.native, m.native_provider)


def text_response(text: str, provider: str = "fake") -> LLMResponse:
    return LLMResponse(text, [], "end", Usage(100, 20), "fake-model", provider)


def tool_response(*calls: ToolCall, provider: str = "fake") -> LLMResponse:
    return LLMResponse("", list(calls), "tool_use", Usage(100, 20), "fake-model", provider)


class MemoryStore:
    """Vector store over the real policy corpus, with the same role pre-filter semantics."""

    def __init__(self, embedder: HashingEmbedder, docs_dir: Path = ROOT / "data/policies"):
        self._embedder = embedder
        self._rows: list[tuple[RetrievedChunk, list[str], np.ndarray]] = []
        self._docs_dir = docs_dir
        self.searches: list[list[str]] = []

    async def load(self) -> MemoryStore:
        next_id = 1
        for path in sorted(self._docs_dir.glob("*.md")):
            doc = load_document(path)
            chunks = chunk_document(doc)
            vectors = await self._embedder.embed([c.embedding_text(doc.title) for c in chunks])
            for c, v in zip(chunks, vectors, strict=True):
                rc = RetrievedChunk(next_id, doc.doc_id, doc.title, c.heading, c.text, 0.0, 0.0, doc.version)
                self._rows.append((rc, doc.allowed_roles, np.asarray(v)))
                next_id += 1
        return self

    def add(self, chunk: RetrievedChunk, roles: list[str], vector: list[float]) -> None:
        self._rows.append((chunk, roles, np.asarray(vector)))

    async def search(self, embedding, query_text, roles, k):
        self.searches.append(list(roles))
        q = np.asarray(embedding)
        scored = []
        for chunk, allowed, vec in self._rows:
            if not set(allowed) & set(roles):
                continue
            sim = float(q @ vec)
            if sim >= 0.1:
                scored.append(RetrievedChunk(**{**vars(chunk), "similarity": sim, "score": sim}))
        return sorted(scored, key=lambda c: -c.similarity)[:k]


class MemoryCache:
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.entries: list[tuple[str, np.ndarray, CachedAnswer]] = []

    async def scope(self, roles, prompt_version):
        return scope_key(roles, "1", prompt_version)

    async def lookup(self, scope, embedding):
        q = np.asarray(embedding)
        best = None
        for s, vec, entry in self.entries:
            sim = float(q @ vec)
            if s == scope and sim >= self.threshold and (best is None or sim > best[0]):
                best = (sim, entry)
        if best is None:
            return None
        return CachedAnswer(**{**vars(best[1]), "similarity": best[0]})

    async def store(self, scope, embedding, entry):
        self.entries.append((scope, np.asarray(embedding), entry))


def tool(name: str, schema: dict[str, Any] | None = None) -> RegisteredTool:
    server, _, remote = name.partition("__")
    spec = ToolSpec(name, f"{remote} tool", schema or {"type": "object", "properties": {}})
    return RegisteredTool(name, server, remote, spec, rbac().tool_policy(name))


class MemoryTools:
    def __init__(self, tools: list[RegisteredTool], results: dict[str, str] | None = None):
        self._tools = tools
        self._results = results or {}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def list_tools(self, principal):
        policy = rbac()
        return [t for t in self._tools if policy.can_use_tool(principal, t.name)]

    async def call(self, principal, tool, arguments):
        self.calls.append((principal.username, tool.name, arguments))
        return self._results.get(tool.name, '{"ok": true}'), False


class MemorySessions:
    def __init__(self):
        self.conversations: dict[str, Conversation] = {}
        self.actions: dict[tuple[str, str], PendingAction] = {}

    async def get_or_create(self, conversation_id, user_sub):
        if conversation_id:
            conv = await self.get(conversation_id, user_sub)
            if conv is None:
                raise ConversationNotFound(conversation_id)
            return conv
        conv = Conversation(str(uuid.uuid4()), user_sub)
        self.conversations[conv.id] = conv
        return conv

    async def get(self, conversation_id, user_sub):
        conv = self.conversations.get(conversation_id)
        return conv if conv and conv.user_sub == user_sub else None

    async def append(self, conversation, role, content):
        conversation.history.append((role, content))

    async def save_vault(self, conversation):
        pass

    async def put_action(self, action):
        self.actions[(action.user_sub, action.id)] = action

    async def pop_action(self, user_sub, action_id):
        return self.actions.pop((user_sub, action_id), None)

    async def allow_request(self, user_sub, limit_per_minute):
        return True


class MemoryPromptLog:
    def __init__(self):
        self.records: list[PromptLogRecord] = []

    def write(self, record):
        self.records.append(record)

    async def drain(self):
        pass


async def build_assistant(
    provider: ScriptedProvider,
    *,
    tools: MemoryTools | None = None,
    cache: MemoryCache | None = None,
    classifier=None,
    **setting_overrides,
) -> tuple[Assistant, dict[str, Any]]:
    s = settings(**setting_overrides)
    embedder = HashingEmbedder(s.embedding_dim)
    store = await MemoryStore(embedder).load()
    parts = {
        "provider": provider,
        "store": store,
        "tools": tools or MemoryTools([]),
        "sessions": MemorySessions(),
        "log": MemoryPromptLog(),
        "cache": cache,
    }
    assistant = Assistant(
        settings=s,
        persona=Persona.load(s.persona_file),
        rbac=rbac(),
        llm=ProviderRegistry(s, {"fake": provider}),
        router=ModelRouter(ModelRef("fake", "small-model"), ModelRef("fake", "frontier-model"), s.router_threshold),
        guardrails=Guardrails(s, classifier),
        embedder=embedder,
        store=store,
        tools=parts["tools"],
        sessions=parts["sessions"],
        prompt_log=parts["log"],
        pricing=Pricing({}),
        cache=cache,
        today=lambda: dt.date(2026, 10, 26),
    )
    return assistant, parts
