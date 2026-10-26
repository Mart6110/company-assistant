"""Jira MCP server: IT tickets filed and read as the signed-in employee.

    python -m assistant.mcp_servers.jira --port 8102

JIRA_BACKEND=cloud talks to Jira Cloud REST v3 with an API token; the ticket's reporter is
set to the employee's Jira account (looked up by email), so "my tickets" means theirs.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal, Protocol

import httpx2
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from assistant.auth.rbac import RBACPolicy
from assistant.config import Settings
from assistant.mcp_servers.common import IntegrationSettings, ServerContext, build_server, run

Priority = Literal["low", "medium", "high", "critical"]
Category = Literal["hardware", "software", "access", "network", "other"]
_JIRA_PRIORITY = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Highest"}


class JiraBackend(Protocol):
    async def create(
        self, *, reporter_email: str, summary: str, description: str, priority: str, category: str
    ) -> dict[str, Any]: ...

    async def issues_for(self, reporter_email: str, limit: int = 20) -> list[dict[str, Any]]: ...

    async def get(self, key: str) -> dict[str, Any] | None: ...


class DemoJiraBackend:
    def __init__(self, project_key: str = "IT"):
        self._project = project_key
        self._issues: dict[str, dict[str, Any]] = {}
        self._ids = itertools.count(1001)

    async def create(self, *, reporter_email, summary, description, priority, category) -> dict[str, Any]:
        key = f"{self._project}-{next(self._ids)}"
        self._issues[key] = {
            "key": key,
            "summary": summary,
            "description": description,
            "priority": _JIRA_PRIORITY[priority],
            "category": category,
            "status": "Open",
            "reporter_email": reporter_email,
        }
        return {"key": key, "status": "Open", "url": f"https://jira.example/browse/{key}"}

    async def issues_for(self, reporter_email: str, limit: int = 20) -> list[dict[str, Any]]:
        return [i for i in self._issues.values() if i["reporter_email"] == reporter_email][-limit:]

    async def get(self, key: str) -> dict[str, Any] | None:
        return self._issues.get(key)


def _adf(text: str) -> dict[str, Any]:
    """Plain text -> Atlassian Document Format, one paragraph per line."""
    lines = [line for line in text.splitlines() if line.strip()] or [""]
    return {
        "type": "doc",
        "version": 1,
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": line}]} for line in lines],
    }


class JiraCloudBackend:
    def __init__(self, base_url: str, email: str, api_token: str, project_key: str, issue_type: str):
        self._http = httpx2.AsyncClient(
            base_url=base_url.rstrip("/"),
            auth=httpx2.BasicAuth(email, api_token),
            headers={"Accept": "application/json"},
            timeout=20,
        )
        self._base = base_url.rstrip("/")
        self._project = project_key
        self._issue_type = issue_type
        self._account_ids: dict[str, str | None] = {}

    async def _account_id(self, email: str) -> str | None:
        if email not in self._account_ids:
            resp = await self._http.get("/rest/api/3/user/search", params={"query": email})
            resp.raise_for_status()
            users = [u for u in resp.json() if u.get("accountType") == "atlassian"]
            self._account_ids[email] = users[0]["accountId"] if users else None
        return self._account_ids[email]

    async def create(self, *, reporter_email, summary, description, priority, category) -> dict[str, Any]:
        account_id = await self._account_id(reporter_email)
        fields: dict[str, Any] = {
            "project": {"key": self._project},
            "issuetype": {"name": self._issue_type},
            "summary": summary,
            "priority": {"name": _JIRA_PRIORITY[priority]},
            "labels": ["it-assistant", category],
            "description": _adf(f"{description}\n\nFiled via the company assistant for {reporter_email}."),
        }
        if account_id:
            fields["reporter"] = {"id": account_id}
        resp = await self._http.post("/rest/api/3/issue", json={"fields": fields})
        if resp.status_code >= 400:
            raise ToolError(f"Jira rejected the ticket ({resp.status_code}): {resp.text[:300]}")
        key = resp.json()["key"]
        return {"key": key, "status": "Open", "url": f"{self._base}/browse/{key}"}

    async def issues_for(self, reporter_email: str, limit: int = 20) -> list[dict[str, Any]]:
        account_id = await self._account_id(reporter_email)
        if account_id is None:
            return []
        resp = await self._http.get(
            "/rest/api/3/search/jql",
            params={
                "jql": f'project = "{self._project}" AND reporter = "{account_id}" ORDER BY created DESC',
                "fields": "summary,status,priority,created",
                "maxResults": limit,
            },
        )
        resp.raise_for_status()
        return [
            {
                "key": i["key"],
                "summary": i["fields"]["summary"],
                "status": i["fields"]["status"]["name"],
                "priority": (i["fields"].get("priority") or {}).get("name"),
                "created": i["fields"]["created"],
            }
            for i in resp.json().get("issues", [])
        ]

    async def get(self, key: str) -> dict[str, Any] | None:
        resp = await self._http.get(f"/rest/api/3/issue/{key}", params={"fields": "summary,status,priority,reporter"})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        f = resp.json()["fields"]
        return {
            "key": key,
            "summary": f["summary"],
            "status": f["status"]["name"],
            "priority": (f.get("priority") or {}).get("name"),
            "reporter_account_id": (f.get("reporter") or {}).get("accountId"),
            "reporter_email": (f.get("reporter") or {}).get("emailAddress"),
        }

    async def owns(self, issue: dict[str, Any], email: str) -> bool:
        return issue.get("reporter_email") == email or issue.get("reporter_account_id") == await self._account_id(email)


def _backend(integrations: IntegrationSettings) -> JiraBackend:
    if integrations.jira_backend == "cloud":
        if not (integrations.jira_base_url and integrations.jira_email and integrations.jira_api_token):
            raise RuntimeError("JIRA_BACKEND=cloud needs JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN")
        return JiraCloudBackend(
            integrations.jira_base_url,
            integrations.jira_email,
            integrations.jira_api_token,
            integrations.jira_project_key,
            integrations.jira_issue_type,
        )
    return DemoJiraBackend(integrations.jira_project_key)


def create_server(
    settings: Settings, integrations: IntegrationSettings, public_url: str, backend: JiraBackend | None = None
) -> MCPServer:
    ctx = ServerContext(settings, RBACPolicy.load(settings.rbac_file))
    jira = backend or _backend(integrations)
    server = build_server("jira", "IT support tickets for the signed-in employee.", public_url, settings)

    def _email(p) -> str:
        if not p.email:
            raise ToolError("the signed-in account has no email address")
        return p.email

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
    async def create_it_ticket(
        summary: str, description: str, priority: Priority = "medium", category: Category = "other"
    ) -> dict[str, Any]:
        """Open an IT support ticket on behalf of the signed-in employee.
        summary: one line, under 120 characters. description: what happened, what they tried, impact."""
        p = ctx.require("it:ticket:create")
        if not summary.strip() or len(summary) > 120:
            raise ToolError("summary must be 1-120 characters")
        return await jira.create(
            reporter_email=_email(p),
            summary=summary.strip(),
            description=description[:5000],
            priority=priority,
            category=category,
        )

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def get_my_tickets() -> dict[str, Any]:
        """List IT tickets the signed-in employee has filed, newest first."""
        p = ctx.require("it:ticket:read_own")
        return {"tickets": await jira.issues_for(_email(p))}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def get_ticket(key: str) -> dict[str, Any]:
        """Get the status of one IT ticket by key, e.g. IT-1042."""
        p = ctx.require("it:ticket:read_own")
        issue = await jira.get(key.strip().upper())
        if issue is None:
            raise ToolError(f"ticket {key} not found")
        owns = issue.get("reporter_email") == p.email
        if isinstance(jira, JiraCloudBackend):
            owns = await jira.owns(issue, _email(p))
        if not owns and not p.has("it:ticket:read_all"):
            # Same answer as a missing ticket: don't confirm that someone else's ticket exists.
            raise ToolError(f"ticket {key} not found")
        return {k: v for k, v in issue.items() if not k.startswith("reporter_")}

    return server


if __name__ == "__main__":
    run(create_server, default_port=8102)
