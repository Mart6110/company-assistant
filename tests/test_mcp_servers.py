"""The MCP servers over real streamable HTTP (in-process ASGI), through the gateway, with
token validation on the server side: identity in, permissions enforced on both sides."""

import datetime as dt
from contextlib import AsyncExitStack, asynccontextmanager

import httpx2
from mcp.server.transport_security import TransportSecuritySettings

from assistant.auth.tokens import DevTokenValidator, principal_from_claims
from assistant.mcp_servers import hr, jira, slack
from assistant.mcp_servers.common import IntegrationSettings
from assistant.tools.gateway import MCPServerConfig, MCPToolGateway, TokenBroker
from tests.fakes import rbac, settings

S = settings()
VALIDATOR = DevTokenValidator(S.dev_jwt_secret, S.oidc_audience)


def user(username, roles, email=None):
    token = VALIDATOR.mint(username=username, roles=roles, email=email)
    return principal_from_claims(VALIDATOR.validate(token), rbac(), S.oidc_roles_claim, token=token)


@asynccontextmanager
async def gateway_for(**servers):
    async with AsyncExitStack() as stack:
        apps = {}
        for name, server in servers.items():
            app = server.streamable_http_app(
                transport_security=TransportSecuritySettings(allowed_hosts=[f"{name}.test"], allowed_origins=[])
            )
            await stack.enter_async_context(app.router.lifespan_context(app))
            apps[f"{name}.test"] = app

        def factory(headers, timeout):
            class Router(httpx2.AsyncBaseTransport):
                async def handle_async_request(self, request):
                    return await httpx2.ASGITransport(apps[request.url.host]).handle_async_request(request)

            return httpx2.AsyncClient(transport=Router(), headers=headers, timeout=timeout)

        configs = [MCPServerConfig(name, f"http://{name}.test/mcp") for name in servers]
        yield MCPToolGateway(configs, rbac(), TokenBroker(S), http_factory=factory)


def servers(**backends):
    integrations = IntegrationSettings(_env_file=None)
    return {
        "hr": hr.create_server(S, integrations, "http://hr.test/mcp", today=lambda: dt.date(2026, 10, 26)),
        "jira": jira.create_server(S, integrations, "http://jira.test/mcp"),
        "slack": slack.create_server(S, integrations, "http://slack.test/mcp", backend=backends.get("slack")),
    }


async def test_tool_catalog_is_filtered_by_role():
    async with gateway_for(**servers()) as gw:
        employee = {t.name for t in await gw.list_tools(user("alice", ["employee"]))}
        manager = {t.name for t in await gw.list_tools(user("mark", ["manager"]))}
    assert "hr__submit_leave_request" in employee and "jira__create_it_ticket" in employee
    assert "hr__list_team_leave_requests" not in employee
    assert "hr__list_team_leave_requests" in manager


async def test_tools_act_as_the_caller():
    async with gateway_for(**servers()) as gw:
        alice = user("alice", ["employee"])
        tools = {t.name: t for t in await gw.list_tools(alice)}
        text, err = await gw.call(
            alice,
            tools["hr__submit_leave_request"],
            {"leave_type": "annual", "start_date": "2026-11-09", "end_date": "2026-11-13"},
        )
        assert not err and '"employee": "alice"' in text and '"working_days": 5' in text
        text, err = await gw.call(alice, tools["hr__get_leave_balance"], {})
        assert '"annual": 13.0' in text

        text, err = await gw.call(
            alice, tools["jira__create_it_ticket"], {"summary": "Laptop fan is loud", "description": "Since Monday"}
        )
        assert not err and "IT-1001" in text
        bob = user("bob", ["employee"])
        text, err = await gw.call(bob, tools["jira__get_ticket"], {"key": "IT-1001"})
        assert err and "not found" in text  # someone else's ticket


async def test_server_enforces_permissions_even_if_gateway_is_bypassed():
    async with gateway_for(**servers()) as gw:
        mark = user("mark", ["manager"])
        team_tool = next(t for t in await gw.list_tools(mark) if t.name == "hr__list_team_leave_requests")
        text, err = await gw.call(user("alice", ["employee"]), team_tool, {})
    assert err and "permission denied" in text


async def test_invalid_token_is_rejected_by_server():
    async with gateway_for(**servers()) as gw:
        forged = DevTokenValidator("wrong-secret", S.oidc_audience).mint(username="eve", roles=["admin"])
        eve = principal_from_claims(
            VALIDATOR.validate(VALIDATOR.mint(username="eve", roles=["admin"])), rbac(), S.oidc_roles_claim, token=forged
        )
        assert await gw.list_tools(eve) == []  # every server answered 401


async def test_slack_channel_allowlist():
    demo = slack.DemoSlackBackend()
    async with gateway_for(**servers(slack=demo)) as gw:
        alice = user("alice", ["employee"])
        post = next(t for t in await gw.list_tools(alice) if t.name == "slack__post_message")
        _, err = await gw.call(alice, post, {"channel": "#random-exec-channel", "text": "hi"})
        assert err
        _, err = await gw.call(alice, post, {"channel": "#it-helpdesk", "text": "VPN down?"})
        assert not err
    assert demo.messages == [{"channel": "#it-helpdesk", "text": "*Alice* (via the company assistant):\nVPN down?"}]
