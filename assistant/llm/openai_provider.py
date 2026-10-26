from __future__ import annotations

import json
from typing import Any

import openai

from assistant.llm.base import LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage

_STOP_REASONS = {
    "stop": "end",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class OpenAIProvider:
    """OpenAI, Azure OpenAI, or any OpenAI-compatible endpoint, via Chat Completions."""

    def __init__(self, client: openai.AsyncOpenAI, *, name: str = "openai"):
        self._client = client
        self.name = name

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
            "messages": [{"role": "system", "content": system}, *self._to_wire(messages)],
            "max_completion_tokens": max_tokens,
        }
        if tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
                }
                for t in tools
            ]
        if effort:
            params["reasoning_effort"] = effort
        if json_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": json_schema, "strict": True},
            }

        try:
            resp = await self._client.chat.completions.create(**params)
        except openai.RateLimitError as e:
            raise LLMError(f"rate limited: {e.message}", provider=self.name, retryable=True) from e
        except openai.APIStatusError as e:
            raise LLMError(e.message, provider=self.name, retryable=e.status_code >= 500) from e
        except openai.APIConnectionError as e:
            raise LLMError(f"connection error: {e}", provider=self.name, retryable=True) from e
        return self._from_wire(resp)

    def _to_wire(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                if m.native is not None and m.native_provider == self.name:
                    out.append(m.native)
                    continue
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in m.tool_calls
                    ]
                out.append(msg)
            else:
                for r in m.tool_results:
                    content = f"ERROR: {r.content}" if r.is_error else r.content
                    out.append({"role": "tool", "tool_call_id": r.call_id, "content": content})
        return out

    def _from_wire(self, resp: Any) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        native_calls: list[dict[str, Any]] = []
        for tc in msg.tool_calls or []:
            if tc.type != "function":
                continue
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None
            valid = isinstance(args, dict)
            calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args if valid else {}, invalid_arguments=not valid)
            )
            native_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
            )
        native: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if native_calls:
            native["tool_calls"] = native_calls

        stop = _STOP_REASONS.get(choice.finish_reason or "", "other")
        if getattr(msg, "refusal", None):
            stop = "refusal"
        u = resp.usage
        cached = 0
        if u is not None and getattr(u, "prompt_tokens_details", None) is not None:
            cached = u.prompt_tokens_details.cached_tokens or 0
        return LLMResponse(
            text=msg.content or "",
            tool_calls=calls,
            stop_reason=stop,
            usage=Usage(
                input_tokens=(u.prompt_tokens if u else 0) or 0,
                output_tokens=(u.completion_tokens if u else 0) or 0,
                cache_read_tokens=cached,
            ),
            model=resp.model,
            provider=self.name,
            native=native,
        )
