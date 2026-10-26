"""Mint a development token (AUTH_MODE=dev only).

    python -m assistant.auth.devtoken alice --roles employee
    python -m assistant.auth.devtoken mark --roles manager
"""

from __future__ import annotations

import argparse

from assistant.auth.tokens import DevTokenValidator
from assistant.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("username")
    parser.add_argument("--roles", default="employee", help="comma-separated, e.g. employee,manager")
    parser.add_argument("--email")
    parser.add_argument("--name")
    args = parser.parse_args()

    settings = get_settings()
    if settings.environment == "production":
        raise SystemExit("refusing to mint dev tokens with ENVIRONMENT=production")
    validator = DevTokenValidator(settings.dev_jwt_secret, settings.oidc_audience)
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    print(validator.mint(username=args.username, roles=roles, email=args.email, name=args.name))


if __name__ == "__main__":
    main()
