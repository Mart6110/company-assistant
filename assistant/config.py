from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_CANARY_SEED = "change-me-canary-seed"


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (or `.env`).

    Provider credentials are read by the provider SDKs themselves:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, AZURE_OPENAI_API_KEY.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"

    # --- Models: "<provider>:<model>", provider is one of anthropic | openai | azure.
    # For azure the model is the deployment name.
    small_model: str = "anthropic:claude-haiku-5-5"
    frontier_model: str = "anthropic:claude-opus-5-5"
    classifier_model: str | None = None  # input classifier; defaults to small_model
    judge_model: str | None = None  # eval LLM-judge; defaults to frontier_model
    # Effort maps to Claude `output_config.effort` / OpenAI `reasoning_effort`.
    # Leave empty for OpenAI models that don't take reasoning_effort.
    small_effort: str | None = "low"
    frontier_effort: str | None = "high"
    max_output_tokens: int = 16000
    anthropic_fallbacks: bool = True  # server-side refusal fallback on models that support it
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-10-21"

    # --- Embeddings: "openai:<model>", "azure:<deployment>", or "hashing" (offline, tests only).
    embedding_model: str = "openai:text-embedding-3-small"
    embedding_dim: int = 1536

    # --- Infrastructure
    database_url: str = "postgresql://assistant:assistant@localhost:5432/assistant"
    redis_url: str = "redis://localhost:6379/0"

    # --- Identity. "dev" accepts HS256 tokens minted by `python -m assistant.auth.devtoken`.
    auth_mode: Literal["oidc", "dev"] = "oidc"
    oidc_issuer: str = "http://localhost:8080/realms/company"
    oidc_audience: str = "assistant-api"
    oidc_jwks_url: str | None = None  # discovered from the issuer when unset
    oidc_roles_claim: str = "realm_access.roles"
    dev_jwt_secret: str = "dev-only-insecure-secret-change-me"
    # How the user's identity reaches MCP servers: forward their access token as-is, or
    # exchange it (RFC 8693) for a token scoped to each server's audience.
    mcp_token_mode: Literal["forward", "exchange"] = "forward"
    oidc_token_url: str | None = None
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None

    # --- Config files
    persona_file: str = "config/persona.yaml"
    rbac_file: str = "config/rbac.yaml"
    mcp_servers_file: str = "config/mcp_servers.yaml"
    pricing_file: str = "config/pricing.yaml"

    # --- RAG
    rag_top_k: int = 5
    rag_min_similarity: float = 0.25

    # --- Semantic cache
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.93
    semantic_cache_ttl_seconds: int = 86400

    # --- Guardrails
    guardrails_llm_classifier: bool = True
    injection_block_threshold: float = 0.7  # heuristic score that blocks outright
    injection_flag_threshold: float = 0.4  # suspicious: the classifier decides, or block if it's unavailable
    untrusted_content_threshold: float = 0.5
    max_input_chars: int = 8000
    canary_seed: str = _DEFAULT_CANARY_SEED

    # --- Routing / agent loop
    router_threshold: float = 0.5
    max_tool_iterations: int = 6
    history_turns: int = 10
    pending_action_ttl_seconds: int = 900
    rate_limit_per_minute: int = 30

    # --- Observability. The OTLP exporter reads OTEL_EXPORTER_OTLP_ENDPOINT itself.
    tracing_enabled: bool = True
    otel_service_name: str = "company-assistant"

    @model_validator(mode="after")
    def _no_published_defaults_in_production(self) -> Settings:
        # The defaults are in a public repository, so they are known values.
        if self.environment == "production" and self.canary_seed == _DEFAULT_CANARY_SEED:
            raise ValueError("set CANARY_SEED to a random value when ENVIRONMENT=production")
        return self

    @property
    def classifier_model_spec(self) -> str:
        return self.classifier_model or self.small_model

    @property
    def judge_model_spec(self) -> str:
        return self.judge_model or self.frontier_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
