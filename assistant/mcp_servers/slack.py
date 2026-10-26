"""Slack MCP server: post to an allow-listed set of channels, attributed to the employee.

    python -m assistant.mcp_servers.slack --port 8103
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx2
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from assistant.auth.rbac import RBACPolicy
from assistant.config import Settings
from assistant.mcp_servers.common import IntegrationSettings, ServerContext, build_server, run


class SlackBackend(Protocol):
    async def post(self, channel: str, text: str) -> dict[str, Any]: ...


class DemoSlackBackend:
    def __init__(self):
        self.messages: list[dict[str, Any]] = []

    async def post(self, channel: str, text: str) -> dict[str, Any]:
        self.messages.append({"channel": channel, "text": text})
        return {"ok": True, "channel": channel, "ts": f"{len(self.messages)}.000"}


class SlackWebBackend:
    def __init__(self, bot_token: str):
        self._http = httpx2.AsyncClient(
            base_url="https://slack.com/api", headers={"Authorization": f"Bearer {bot_token}"}, timeout=15
        )

    async def post(self, channel: str, text: str) -> dict[str, Any]:
        resp = await self._http.post("/chat.postMessage", json={"channel": channel, "text": text})
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise ToolError(f"Slack rejected the message: {data.get('error', 'unknown_error')}")
        return {"ok": True, "channel": data["channel"], "ts": data["ts"]}


def create_server(
    settings: Settings, integrations: IntegrationSettings, public_url: str, backend: SlackBackend | None = None
) -> MCPServer:
    ctx = ServerContext(settings, RBACPolicy.load(settings.rbac_file))
    allowed = [c.strip() for c in integrations.slack_allowed_channels.split(",") if c.strip()]
    if backend is None:
        if integrations.slack_backend == "web":
            if not integrations.slack_bot_token:
                raise RuntimeError("SLACK_BACKEND=web needs SLACK_BOT_TOKEN")
            backend = SlackWebBackend(integrations.slack_bot_token)
        else:
            backend = DemoSlackBackend()
    slack = backend
    server = build_server(
        "slack", f"Post messages to these Slack channels: {', '.join(allowed)}.", public_url, settings
    )

    @server.tool(
        description=(
            "Post a message to a Slack channel on behalf of the signed-in employee. "
            f"channel must be one of: {', '.join(allowed)}."
        ),
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True),
    )
    async def post_message(channel: str, text: str) -> dict[str, Any]:
        p = ctx.require("slack:post")
        if channel not in allowed:
            raise ToolError(f"posting to {channel} is not allowed; allowed channels: {', '.join(allowed)}")
        if not text.strip() or len(text) > 3000:
            raise ToolError("text must be 1-3000 characters")
        return await slack.post(channel, f"*{p.name or p.username}* (via the company assistant):\n{text}")

    return server


if __name__ == "__main__":
    run(create_server, default_port=8103)
