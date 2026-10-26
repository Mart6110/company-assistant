from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace

from assistant.config import Settings
from assistant.llm.base import LLMError, LLMProvider, LLMResponse, Message, ModelRef, ToolSpec

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class ProviderRegistry:
    """Builds provider adapters lazily and wraps every completion in a GenAI span."""

    def __init__(self, settings: Settings, providers: dict[str, LLMProvider] | None = None):
        self._settings = settings
        self._providers: dict[str, LLMProvider] = dict(providers or {})

    def get(self, name: str) -> LLMProvider:
        if name not in self._providers:
            self._providers[name] = self._build(name)
        return self._providers[name]

    def _build(self, name: str) -> LLMProvider:
        if name == "anthropic":
            from assistant.llm.anthropic_provider import AnthropicProvider

            return AnthropicProvider(fallbacks=self._settings.anthropic_fallbacks)
        if name == "openai":
            import openai

            from assistant.llm.openai_provider import OpenAIProvider

            return OpenAIProvider(openai.AsyncOpenAI(), name="openai")
        if name == "azure":
            import openai

            from assistant.llm.openai_provider import OpenAIProvider

            client = openai.AsyncAzureOpenAI(
                azure_endpoint=self._settings.azure_openai_endpoint,
                api_version=self._settings.azure_openai_api_version,
            )
            return OpenAIProvider(client, name="azure")
        raise ValueError(f"unknown LLM provider {name!r} (expected anthropic, openai or azure)")

    async def complete(
        self,
        ref: ModelRef,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
        purpose: str = "chat",
    ) -> LLMResponse:
        with tracer.start_as_current_span(f"chat {ref.model}") as span:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", ref.provider)
            span.set_attribute("gen_ai.request.model", ref.model)
            span.set_attribute("assistant.llm.purpose", purpose)
            if ref.effort:
                span.set_attribute("assistant.llm.effort", ref.effort)
            try:
                resp = await self.get(ref.provider).complete(
                    model=ref.model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens or self._settings.max_output_tokens,
                    effort=ref.effort,
                    json_schema=json_schema,
                )
            except LLMError as e:
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise
            except Exception as e:
                # Missing credentials and other client-side failures surface as plain SDK or
                # Python exceptions; normalize them at the provider boundary.
                log.exception("%s provider failed before returning a response", ref.provider)
                span.record_exception(e)
                span.set_status(trace.StatusCode.ERROR, str(e))
                raise LLMError(f"{ref.provider}: {type(e).__name__}: {e}", provider=ref.provider, retryable=False) from e
            span.set_attribute("gen_ai.response.model", resp.model)
            span.set_attribute("gen_ai.response.finish_reasons", [resp.stop_reason])
            span.set_attribute("gen_ai.usage.input_tokens", resp.usage.input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", resp.usage.output_tokens)
            span.set_attribute("assistant.llm.cache_read_tokens", resp.usage.cache_read_tokens)
            if resp.tool_calls:
                span.set_attribute("assistant.llm.tool_calls", [c.name for c in resp.tool_calls])
            return resp
