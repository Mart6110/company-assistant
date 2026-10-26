"""Tool gateway over MCP servers (HR, Jira, Slack, ...).

- Tools are namespaced `<server>__<tool>` and exposed only if config/rbac.yaml has a
  policy for them that the caller's permissions satisfy (deny by default).
- Every MCP request carries the *caller's* identity: their access token is forwarded, or
  exchanged (RFC 8693) for one whose audience is that MCP server. Each server validates it
  and enforces permissions again, so a bug here cannot grant access on its own.
- Trace context propagates to the servers through MCP `_meta` (done by the MCP SDK).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx2
import yaml
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Implementation

from assistant.auth.rbac import Principal, RBACPolicy, ToolPolicy
from assistant.config import Settings
from assistant.llm import ToolSpec

log = logging.getLogger(__name__)

SEPARATOR = "__"


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    url: str
    audience: str | None = None  # token-exchange audience; defaults to the server name


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    server: str
    remote_name: str
    spec: ToolSpec
    policy: ToolPolicy


class ToolGateway(Protocol):
    async def list_tools(self, principal: Principal) -> list[RegisteredTool]: ...

    async def call(self, principal: Principal, tool: RegisteredTool, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Returns (result text, is_error)."""
        ...


def load_server_configs(path: str | Path) -> list[MCPServerConfig]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return [MCPServerConfig(**s) for s in data.get("servers", [])]


class TokenBroker:
    """Produces the bearer token sent to an MCP server on the caller's behalf."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._cache: dict[tuple[str, str], tuple[str, float]] = {}

    async def token_for(self, principal: Principal, server: MCPServerConfig) -> str:
        if principal.token is None:
            raise PermissionError("no caller token available to pass to MCP servers")
        if self._settings.mcp_token_mode == "forward":
            return principal.token
        audience = server.audience or server.name
        key = (principal.sub, audience)
        cached = self._cache.get(key)
        if cached and cached[1] > time.time() + 30:
            return cached[0]
        token, expires_at = await self._exchange(principal.token, audience)
        self._cache[key] = (token, expires_at)
        return token

    async def _exchange(self, subject_token: str, audience: str) -> tuple[str, float]:
        s = self._settings
        if not (s.oidc_token_url and s.oidc_client_id and s.oidc_client_secret):
            raise RuntimeError("MCP_TOKEN_MODE=exchange needs OIDC_TOKEN_URL, OIDC_CLIENT_ID and OIDC_CLIENT_SECRET")
        async with httpx2.AsyncClient(timeout=10) as http:
            resp = await http.post(
                s.oidc_token_url,
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                    "subject_token": subject_token,
                    "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
                    "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
                    "audience": audience,
                },
                auth=httpx2.BasicAuth(s.oidc_client_id, s.oidc_client_secret),
            )
            resp.raise_for_status()
            body = resp.json()
        return body["access_token"], time.time() + float(body.get("expires_in", 300))


def _result_text(result: Any) -> str:
    if result.structured_content is not None:
        return json.dumps(result.structured_content, default=str)
    return "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")


class MCPToolGateway:
    def __init__(
        self,
        servers: list[MCPServerConfig],
        rbac: RBACPolicy,
        tokens: TokenBroker,
        catalog_ttl_seconds: float = 300,
        call_timeout_seconds: float = 60,
        http_factory: Callable[[dict[str, str], httpx2.Timeout], httpx2.AsyncClient] | None = None,
    ):
        self._servers = {s.name: s for s in servers}
        self._rbac = rbac
        self._tokens = tokens
        self._ttl = catalog_ttl_seconds
        self._timeout = call_timeout_seconds
        self._http_factory = http_factory or (lambda headers, timeout: httpx2.AsyncClient(headers=headers, timeout=timeout))
        # Catalog per role scope: a server may tailor its tool list to the caller.
        self._catalogs: dict[str, tuple[float, list[RegisteredTool]]] = {}
        self._warned_unlisted: set[str] = set()

    @asynccontextmanager
    async def _connect(self, principal: Principal, server: MCPServerConfig) -> AsyncIterator[Client]:
        token = await self._tokens.token_for(principal, server)
        timeout = httpx2.Timeout(30, read=self._timeout)
        async with self._http_factory({"Authorization": f"Bearer {token}"}, timeout) as http:
            transport = streamable_http_client(server.url, http_client=http)
            async with Client(transport, client_info=Implementation(name="company-assistant", version="0.1.0")) as c:
                yield c

    async def _list_server(self, principal: Principal, server: MCPServerConfig) -> list[RegisteredTool]:
        tools: list[RegisteredTool] = []
        async with self._connect(principal, server) as client:
            cursor = None
            while True:
                page = await client.list_tools(cursor=cursor)
                for t in page.tools:
                    name = f"{server.name}{SEPARATOR}{t.name}"
                    policy = self._rbac.tool_policy(name)
                    if policy is None:
                        if name not in self._warned_unlisted:
                            log.warning("MCP tool %s has no RBAC policy and is not exposed", name)
                            self._warned_unlisted.add(name)
                        continue
                    spec = ToolSpec(name=name, description=t.description or t.title or t.name, input_schema=t.input_schema)
                    tools.append(RegisteredTool(name, server.name, t.name, spec, policy))
                cursor = page.next_cursor
                if not cursor:
                    break
        return tools

    async def _catalog(self, principal: Principal) -> list[RegisteredTool]:
        cached = self._catalogs.get(principal.role_scope)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        results = await asyncio.gather(
            *(self._list_server(principal, s) for s in self._servers.values()), return_exceptions=True
        )
        catalog: list[RegisteredTool] = []
        for server, result in zip(self._servers.values(), results, strict=True):
            if isinstance(result, BaseException):
                # One unavailable backend shouldn't take the assistant down; its tools just disappear.
                log.warning("MCP server %s unavailable: %s", server.name, result)
                continue
            catalog.extend(result)
        catalog.sort(key=lambda t: t.name)  # deterministic order keeps the prompt cache prefix stable
        if len(catalog) or not self._servers:
            self._catalogs[principal.role_scope] = (time.monotonic() + self._ttl, catalog)
        return catalog

    async def list_tools(self, principal: Principal) -> list[RegisteredTool]:
        return [t for t in await self._catalog(principal) if self._rbac.can_use_tool(principal, t.name)]

    async def call(self, principal: Principal, tool: RegisteredTool, arguments: dict[str, Any]) -> tuple[str, bool]:
        server = self._servers[tool.server]
        async with self._connect(principal, server) as client:
            result = await client.call_tool(tool.remote_name, arguments)
        return _result_text(result), bool(result.is_error)
