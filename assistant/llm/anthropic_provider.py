from __future__ import annotations

from typing import Any

import anthropic

from assistant.llm.base import LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage

# Models that take the server-side refusal fallback (`fallbacks: "default"`). Haiku has no
# server-side fallback, so it is not listed.
_FALLBACK_MODEL_PREFIXES = ("claude-fable-5", "claude-opus-5", "claude-sonnet-5-5")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STOP_REASONS = {
    "end_turn": "end",
    "stop_sequence": "end",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


class AnthropicProvider:
    """Claude via the official Anthropic SDK.

    Notes for current models (Opus 5.5 / Sonnet 5.5 / Haiku 5.5):
    - thinking is adaptive and always on; depth is controlled with `output_config.effort`.
    - sampling parameters (temperature/top_p/top_k) are rejected, so none are sent.
    - forced `tool_choice` is rejected, so tool choice stays at the default `auto`.
    """

    name = "anthropic"

    def __init__(self, client: anthropic.AsyncAnthropic | None = None, *, fallbacks: bool = True):
        self._client = client or anthropic.AsyncAnthropic()
        self._fallbacks = fallbacks

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
        effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._to_wire(messages),
            # Automatic caching of the longest stable prefix (tools, system, history), so each
            # tool-loop iteration reads the previous iteration's prefix from cache.
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools
            ]
        output_config: dict[str, Any] = {}
        if effort:
            output_config["effort"] = effort
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            params["output_config"] = output_config

        try:
            if self._fallbacks and model.startswith(_FALLBACK_MODEL_PREFIXES):
                resp = await self._client.beta.messages.create(
                    **params, betas=[_FALLBACK_BETA], fallbacks="default"
                )
            else:
                resp = await self._client.messages.create(**params)
        except anthropic.RateLimitError as e:
            raise LLMError(f"rate limited: {e.message}", provider=self.name, retryable=True) from e
        except anthropic.APIStatusError as e:
            raise LLMError(e.message, provider=self.name, retryable=e.status_code >= 500) from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"connection error: {e}", provider=self.name, retryable=True) from e
        return self._from_wire(resp)

    def _to_wire(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                if m.native is not None and m.native_provider == self.name:
                    out.append({"role": "assistant", "content": m.native})
                    continue
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls
                ]
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
            else:
                # Every result for one assistant turn goes back in a single user message.
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": r.call_id,
                                "content": r.content,
                                "is_error": r.is_error,
                            }
                            for r in m.tool_results
                        ],
                    }
                )
        return out

    def _from_wire(self, resp: Any) -> LLMResponse:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=args, invalid_arguments=args is not block.input)
                )
        category = None
        if resp.stop_reason == "refusal" and getattr(resp, "stop_details", None) is not None:
            category = resp.stop_details.category
        u = resp.usage
        return LLMResponse(
            text="".join(texts),
            tool_calls=calls,
            stop_reason=_STOP_REASONS.get(resp.stop_reason, "other"),
            usage=Usage(
                input_tokens=u.input_tokens or 0,
                output_tokens=u.output_tokens or 0,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            ),
            model=resp.model,
            provider=self.name,
            # Content blocks (thinking, text, tool_use, fallback) are replayed unchanged
            # within the turn's tool loop.
            native=resp.content,
            refusal_category=category,
        )
