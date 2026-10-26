"""Shared plumbing for the HR / Jira / Slack MCP servers.

Each server is an OAuth resource server: it verifies the bearer token the assistant
forwards (the employee's own identity), resolves roles with the same RBAC policy, and
checks permissions itself. Tools act *as the employee*, never as a shared service user.
"""

from __future__ import annotations

import argparse
from typing import Literal

import anyio
import uvicorn
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic_settings import BaseSettings, SettingsConfigDict

from assistant.auth.rbac import Principal, RBACPolicy
from assistant.auth.tokens import TokenError, TokenValidator, build_validator, principal_from_claims
from assistant.config import Settings, get_settings
from assistant.observability.tracing import setup_tracing


class IntegrationSettings(BaseSettings):
    """Backend credentials. Only the MCP server containers get these, never the assistant API."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hr_backend: Literal["demo"] = "demo"
    jira_backend: Literal["demo", "cloud"] = "demo"
    jira_base_url: str | None = None  # https://your-site.atlassian.net
    jira_email: str | None = None
    jira_api_token: str | None = None
    jira_project_key: str = "IT"
    jira_issue_type: str = "Task"
    slack_backend: Literal["demo", "web"] = "demo"
    slack_bot_token: str | None = None
    slack_allowed_channels: str = "#it-helpdesk,#hr-questions,#general"


class JWTTokenVerifier:
    """Adapts the assistant's token validators to the MCP SDK's TokenVerifier protocol."""

    def __init__(self, validator: TokenValidator):
        self._validator = validator

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = await anyio.to_thread.run_sync(self._validator.validate, token)
        except TokenError:
            return None
        return AccessToken(
            token=token,
            client_id=claims.get("azp") or claims.get("client_id") or "unknown",
            scopes=(claims.get("scope") or "").split(),
            expires_at=claims.get("exp"),
            subject=claims.get("sub"),
            claims=claims,
        )


class ServerContext:
    def __init__(self, settings: Settings, rbac: RBACPolicy):
        self.settings = settings
        self.rbac = rbac

    def principal(self) -> Principal:
        token = get_access_token()
        if token is None or token.claims is None:
            raise ToolError("unauthenticated")
        return principal_from_claims(token.claims, self.rbac, self.settings.oidc_roles_claim)

    def require(self, permission: str) -> Principal:
        principal = self.principal()
        if not principal.has(permission):
            raise ToolError(f"permission denied: {permission} required")
        return principal


def build_server(name: str, instructions: str, public_url: str, settings: Settings) -> MCPServer:
    validator = build_validator(settings)
    return MCPServer(
        name,
        instructions=instructions,
        version="0.1.0",
        token_verifier=JWTTokenVerifier(validator),
        auth=AuthSettings(
            issuer_url=validator.issuer,
            resource_server_url=public_url,
            # The verifier checks issuer, signature and audience itself.
            validate_token_resource=False,
        ),
    )


def run(create, default_port: int) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--public-url", help="URL clients use, e.g. http://hr-mcp:8101/mcp")
    parser.add_argument("--allowed-hosts", default="", help="extra Host headers to accept, comma-separated")
    args = parser.parse_args()

    settings = get_settings()
    public_url = args.public_url or f"http://localhost:{args.port}/mcp"
    server: MCPServer = create(settings, IntegrationSettings(), public_url)
    if settings.tracing_enabled:
        setup_tracing(f"mcp-{server.name}", settings.environment)
    allowed = {f"localhost:{args.port}", f"127.0.0.1:{args.port}"}
    allowed |= {h.strip() for h in args.allowed_hosts.split(",") if h.strip()}
    app = server.streamable_http_app(
        transport_security=TransportSecuritySettings(allowed_hosts=sorted(allowed), allowed_origins=[]),
        host=args.host,
    )
    uvicorn.run(app, host=args.host, port=args.port)
