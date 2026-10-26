.PHONY: install up down logs ingest token kc-token test test-integration lint eval-offline eval

COMPOSE ?= docker compose
USER ?= alice
ROLES ?= employee

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev,pdf]"

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api hr-mcp jira-mcp slack-mcp

ingest:
	$(COMPOSE) run --rm api python -m assistant.rag.ingest data/policies --prune

# Dev token (AUTH_MODE=dev): make token USER=mark ROLES=manager
token:
	@$(COMPOSE) run --rm -T api python -m assistant.auth.devtoken $(USER) --roles $(ROLES)

# Keycloak token (AUTH_MODE=oidc): make kc-token USER=mark  (password = username in the dev realm)
kc-token:
	@curl -s -d grant_type=password -d client_id=assistant-web -d username=$(USER) -d password=$(USER) \
		http://localhost:8080/realms/company/protocol/openid-connect/token | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])"

lint:
	.venv/bin/ruff check assistant tests evals

test:
	.venv/bin/pytest

test-integration:
	.venv/bin/pytest -m integration

eval-offline:
	.venv/bin/python -m evals.run --offline --gate

eval:
	$(COMPOSE) run --rm api python -m evals.run --online --gate
