import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from assistant.auth.tokens import DevTokenValidator, OIDCValidator, TokenError, principal_from_claims
from tests.fakes import principal, rbac


def test_role_inheritance_and_permissions():
    mark = principal("mark", ("manager",))
    assert mark.roles == {"manager", "employee"}
    assert mark.has("hr:team:read") and mark.has("hr:leave:request")
    assert not mark.has("it:ticket:read_all")


def test_wildcard_permission():
    ada = principal("ada", ("admin",))
    assert ada.has("admin:prompt_logs:read") and ada.has("admin:router:override")
    assert not ada.has("hr:team:read")


def test_unknown_idp_roles_are_ignored():
    p = principal("x", ("employee", "offline_access", "uma_authorization"))
    assert p.roles == {"employee"}


def test_tools_are_deny_by_default():
    policy = rbac()
    alice = principal()
    assert policy.can_use_tool(alice, "hr__submit_leave_request")
    assert not policy.can_use_tool(alice, "hr__list_team_leave_requests")
    assert not policy.can_use_tool(alice, "hr__delete_everything")  # not in rbac.yaml


def test_dev_token_round_trip():
    v = DevTokenValidator("secret", "assistant-api")
    claims = v.validate(v.mint(username="alice", roles=["employee"]))
    p = principal_from_claims(claims, rbac(), "realm_access.roles")
    assert p.username == "alice" and p.roles == {"employee"}


def test_dev_token_rejects_wrong_secret_and_audience():
    good = DevTokenValidator("secret", "assistant-api")
    with pytest.raises(TokenError):
        DevTokenValidator("other", "assistant-api").validate(good.mint(username="a", roles=[]))
    with pytest.raises(TokenError):
        DevTokenValidator("secret", "other-api").validate(good.mint(username="a", roles=[]))


class _StaticJWKS:
    def __init__(self, key):
        self._key = key

    def get_signing_key_from_jwt(self, token):
        return type("K", (), {"key": self._key})()


def _rsa_token(key, **overrides):
    now = int(time.time())
    claims = {
        "iss": "https://idp.example/realms/company",
        "aud": ["assistant-api", "account"],
        "sub": "user-123",
        "iat": now,
        "exp": now + 300,
        "realm_access": {"roles": ["employee", "manager"]},
    }
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256")


def test_oidc_validator_checks_signature_issuer_audience_expiry():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    v = OIDCValidator(
        "https://idp.example/realms/company", "assistant-api", jwks_client=_StaticJWKS(key.public_key())
    )
    assert v.validate(_rsa_token(key))["sub"] == "user-123"
    for bad in (
        {"iss": "https://evil.example"},
        {"aud": "someone-else"},
        {"exp": int(time.time()) - 3600},
    ):
        with pytest.raises(TokenError):
            v.validate(_rsa_token(key, **bad))
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(TokenError):
        v.validate(_rsa_token(other))


def test_production_refuses_published_defaults():
    from assistant.auth.tokens import build_validator
    from tests.fakes import settings

    with pytest.raises(ValueError, match="CANARY_SEED"):
        settings(environment="production", auth_mode="oidc")
    with pytest.raises(RuntimeError, match="AUTH_MODE=dev"):
        build_validator(settings(environment="production", canary_seed="random-value"))
