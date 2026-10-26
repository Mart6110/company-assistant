from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Protocol

import jwt

from assistant.auth.rbac import Principal, RBACPolicy
from assistant.config import Settings


class TokenError(Exception):
    pass


class TokenValidator(Protocol):
    issuer: str

    def validate(self, token: str) -> dict[str, Any]:
        """Return verified claims or raise TokenError. May block on a JWKS fetch."""
        ...


class OIDCValidator:
    """Validates access tokens from any OIDC provider (Keycloak, Entra ID, Okta, Auth0...)."""

    algorithms = ["RS256", "PS256", "ES256"]

    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_url: str | None = None,
        jwks_client: jwt.PyJWKClient | None = None,
    ):
        self.issuer = issuer
        self.audience = audience
        self._jwks_url = jwks_url
        self._jwks = jwks_client

    def _client(self) -> jwt.PyJWKClient:
        if self._jwks is None:
            url = self._jwks_url or self._discover_jwks_url()
            self._jwks = jwt.PyJWKClient(url, cache_keys=True, lifespan=300)
        return self._jwks

    def _discover_jwks_url(self) -> str:
        with urllib.request.urlopen(f"{self.issuer.rstrip('/')}/.well-known/openid-configuration", timeout=5) as r:
            return json.load(r)["jwks_uri"]

    def validate(self, token: str) -> dict[str, Any]:
        try:
            key = self._client().get_signing_key_from_jwt(token).key
            return jwt.decode(
                token,
                key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat", "sub"]},
                leeway=30,
            )
        except jwt.PyJWTError as e:
            raise TokenError(str(e)) from e


class DevTokenValidator:
    """HS256 tokens for local development and evals. Refused in production."""

    issuer = "http://assistant-dev.local"

    def __init__(self, secret: str, audience: str):
        self._secret = secret
        self.audience = audience

    def validate(self, token: str) -> dict[str, Any]:
        try:
            return jwt.decode(
                token,
                self._secret,
                algorithms=["HS256"],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat", "sub"]},
            )
        except jwt.PyJWTError as e:
            raise TokenError(str(e)) from e

    def mint(
        self,
        *,
        username: str,
        roles: list[str],
        email: str | None = None,
        name: str | None = None,
        ttl_seconds: int = 8 * 3600,
    ) -> str:
        now = int(time.time())
        claims = {
            "iss": self.issuer,
            "aud": self.audience,
            "sub": f"dev-{username}",
            "preferred_username": username,
            "email": email or f"{username}@acme.example",
            "name": name or username.title(),
            "given_name": (name or username).split()[0].title(),
            # Same claim shape as Keycloak, so role extraction is identical in both modes.
            "realm_access": {"roles": roles},
            "iat": now,
            "exp": now + ttl_seconds,
        }
        return jwt.encode(claims, self._secret, algorithm="HS256")


def build_validator(settings: Settings) -> TokenValidator:
    if settings.auth_mode == "dev":
        if settings.environment == "production":
            raise RuntimeError("AUTH_MODE=dev is not allowed when ENVIRONMENT=production")
        return DevTokenValidator(settings.dev_jwt_secret, settings.oidc_audience)
    return OIDCValidator(settings.oidc_issuer, settings.oidc_audience, settings.oidc_jwks_url)


def roles_from_claims(claims: dict[str, Any], path: str) -> list[str]:
    node: Any = claims
    for part in path.split("."):
        if not isinstance(node, dict):
            return []
        node = node.get(part)
    if isinstance(node, str):
        return node.split()
    return [r for r in node or [] if isinstance(r, str)]


def principal_from_claims(
    claims: dict[str, Any], rbac: RBACPolicy, roles_claim: str, token: str | None = None
) -> Principal:
    return rbac.principal(
        sub=claims["sub"],
        username=claims.get("preferred_username") or claims["sub"],
        roles=roles_from_claims(claims, roles_claim),
        email=claims.get("email"),
        name=claims.get("name"),
        given_name=claims.get("given_name"),
        token=token,
    )
