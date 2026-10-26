"""HR MCP server: leave balances and leave requests.

    python -m assistant.mcp_servers.hr --port 8101

`DemoHRBackend` keeps state in memory. To connect a real HRIS (Workday, Personio,
BambooHR, ...), implement `HRBackend` against its API and select it in `create_server`.
"""

from __future__ import annotations

import datetime as dt
import itertools
from typing import Any, Literal, Protocol

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from assistant.auth.rbac import RBACPolicy
from assistant.config import Settings
from assistant.mcp_servers.common import IntegrationSettings, ServerContext, build_server, run

LeaveType = Literal["annual", "sick", "parental", "unpaid"]


class HRBackend(Protocol):
    async def balance(self, username: str) -> dict[str, float]: ...

    async def submit(
        self, username: str, leave_type: str, start: dt.date, end: dt.date, days: int, reason: str
    ) -> dict[str, Any]: ...

    async def requests_for(self, username: str) -> list[dict[str, Any]]: ...

    async def team_requests(self, manager: str) -> list[dict[str, Any]]: ...


class DemoHRBackend:
    def __init__(self):
        self._balances: dict[str, dict[str, float]] = {}
        self._requests: list[dict[str, Any]] = []
        self._ids = itertools.count(1001)
        self._manager_of = {"alice": "mark", "ivan": "mark", "hanna": "ada"}

    async def balance(self, username: str) -> dict[str, float]:
        return self._balances.setdefault(username, {"annual": 18.0, "sick": 10.0, "parental": 0.0, "unpaid": 30.0})

    async def submit(
        self, username: str, leave_type: str, start: dt.date, end: dt.date, days: int, reason: str
    ) -> dict[str, Any]:
        balance = await self.balance(username)
        if leave_type in ("annual", "unpaid") and days > balance[leave_type]:
            raise ToolError(f"only {balance[leave_type]:g} {leave_type} days remaining, {days} requested")
        if leave_type in ("annual", "unpaid"):
            balance[leave_type] -= days
        request = {
            "request_id": f"LR-{next(self._ids)}",
            "employee": username,
            "approver": self._manager_of.get(username, "hr-team"),
            "leave_type": leave_type,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "working_days": days,
            "reason": reason,
            "status": "pending_approval",
        }
        self._requests.append(request)
        return request

    async def requests_for(self, username: str) -> list[dict[str, Any]]:
        return [r for r in self._requests if r["employee"] == username]

    async def team_requests(self, manager: str) -> list[dict[str, Any]]:
        return [r for r in self._requests if r["approver"] == manager]


def working_days(start: dt.date, end: dt.date) -> int:
    return sum(1 for i in range((end - start).days + 1) if (start + dt.timedelta(days=i)).weekday() < 5)


def create_server(
    settings: Settings,
    integrations: IntegrationSettings,
    public_url: str,
    backend: HRBackend | None = None,
    today=dt.date.today,
) -> MCPServer:
    ctx = ServerContext(settings, RBACPolicy.load(settings.rbac_file))
    hr: HRBackend = backend or DemoHRBackend()
    server = build_server(
        "hr", "Leave balances and leave requests for the signed-in employee.", public_url, settings
    )

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def get_leave_balance() -> dict[str, Any]:
        """Get the signed-in employee's remaining leave, in days, per leave type."""
        p = ctx.require("hr:leave:read_own")
        return {"employee": p.username, "balances_days": await hr.balance(p.username)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def list_my_leave_requests() -> dict[str, Any]:
        """List the signed-in employee's leave requests and their approval status."""
        p = ctx.require("hr:leave:read_own")
        return {"requests": await hr.requests_for(p.username)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
    async def submit_leave_request(
        leave_type: LeaveType, start_date: dt.date, end_date: dt.date, reason: str = ""
    ) -> dict[str, Any]:
        """Submit a leave request for the signed-in employee. Dates are ISO (YYYY-MM-DD), both inclusive.
        The request goes to their manager for approval."""
        p = ctx.require("hr:leave:request")
        if end_date < start_date:
            raise ToolError("end_date is before start_date")
        if start_date < today() and leave_type != "sick":
            raise ToolError("only sick leave can be registered retroactively")
        days = working_days(start_date, end_date)
        if days == 0:
            raise ToolError("the requested period contains no working days")
        return await hr.submit(p.username, leave_type, start_date, end_date, days, reason[:500])

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def list_team_leave_requests() -> dict[str, Any]:
        """List leave requests awaiting the signed-in manager's approval."""
        p = ctx.require("hr:team:read")
        return {"requests": await hr.team_requests(p.username)}

    return server


if __name__ == "__main__":
    run(create_server, default_port=8101)
