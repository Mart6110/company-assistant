"""Role-based access control.

Roles come from the caller's OAuth token; `config/rbac.yaml` maps them to permissions,
decides which tools each role may see, and which tools need explicit user confirmation.
Document access is decided by role: each policy document lists `allowed_roles`, matched
against the caller's effective (inherited) roles inside the vector query itself.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ToolPolicy:
    permission: str
    confirm: bool = False  # side-effecting: held until the user confirms


@dataclass(frozen=True)
class Principal:
    sub: str
    username: str
    roles: frozenset[str]  # effective roles, inheritance already applied
    permissions: frozenset[str]
    email: str | None = None
    name: str | None = None
    given_name: str | None = None
    token: str | None = field(default=None, repr=False, compare=False)

    def has(self, permission: str) -> bool:
        return any(_grants(granted, permission) for granted in self.permissions)

    @property
    def role_scope(self) -> str:
        """Stable key for everything whose visibility depends on roles (cache entries, tool catalogs)."""
        return hashlib.sha256("|".join(sorted(self.roles)).encode()).hexdigest()[:16]


def _grants(granted: str, required: str) -> bool:
    if granted == "*" or granted == required:
        return True
    return granted.endswith(":*") and required.startswith(granted[:-1])


@dataclass(frozen=True)
class RoleDef:
    permissions: frozenset[str]
    inherits: tuple[str, ...] = ()


class RBACPolicy:
    def __init__(self, roles: dict[str, RoleDef], tools: dict[str, ToolPolicy]):
        self.roles = roles
        self.tools = tools

    @classmethod
    def load(cls, path: str | Path) -> RBACPolicy:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RBACPolicy:
        roles = {
            name: RoleDef(
                permissions=frozenset(spec.get("permissions", [])),
                inherits=tuple(spec.get("inherits", [])),
            )
            for name, spec in (data.get("roles") or {}).items()
        }
        tools = {
            name: ToolPolicy(permission=spec["permission"], confirm=bool(spec.get("confirm", False)))
            for name, spec in (data.get("tools") or {}).items()
        }
        return cls(roles, tools)

    def effective_roles(self, roles: Iterable[str]) -> frozenset[str]:
        """Known roles plus everything they inherit. Unknown IdP roles are ignored."""
        seen: set[str] = set()
        stack = [r for r in roles if r in self.roles]
        while stack:
            role = stack.pop()
            if role in seen:
                continue
            seen.add(role)
            stack.extend(r for r in self.roles[role].inherits if r in self.roles)
        return frozenset(seen)

    def principal(
        self,
        *,
        sub: str,
        username: str,
        roles: Iterable[str],
        email: str | None = None,
        name: str | None = None,
        given_name: str | None = None,
        token: str | None = None,
    ) -> Principal:
        effective = self.effective_roles(roles)
        permissions = frozenset(p for r in effective for p in self.roles[r].permissions)
        return Principal(
            sub=sub,
            username=username,
            roles=effective,
            permissions=permissions,
            email=email,
            name=name,
            given_name=given_name,
            token=token,
        )

    def tool_policy(self, tool_name: str) -> ToolPolicy | None:
        return self.tools.get(tool_name)

    def can_use_tool(self, principal: Principal, tool_name: str) -> bool:
        # Deny by default: a tool with no policy entry is never exposed or executed.
        policy = self.tools.get(tool_name)
        return policy is not None and principal.has(policy.permission)
