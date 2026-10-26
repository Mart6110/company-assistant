"""Provider-neutral types for chat completion with tool calling.

Every provider adapter converts these to and from its own wire format, so the rest of
the assistant never imports a provider SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

StopReason = Literal["end", "tool_use", "max_tokens", "refusal", "other"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    invalid_arguments: bool = False  # the model emitted arguments that were not a JSON object


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # Provider-native assistant payload (e.g. Claude content blocks including thinking).
    # The provider that produced it replays it verbatim, which keeps the tool loop
    # append-only; any other provider falls back to the neutral fields above.
    native: Any = None
    native_provider: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: StopReason
    usage: Usage
    model: str
    provider: str
    native: Any = None
    refusal_category: str | None = None

    def as_message(self) -> Message:
        return Message(
            role="assistant",
            content=self.text,
            tool_calls=list(self.tool_calls),
            native=self.native,
            native_provider=self.provider,
        )


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str
    effort: str | None = None

    @classmethod
    def parse(cls, spec: str, effort: str | None = None) -> ModelRef:
        provider, sep, model = spec.partition(":")
        if not sep or not provider or not model:
            raise ValueError(f"model spec must look like '<provider>:<model>', got {spec!r}")
        return cls(provider=provider.strip().lower(), model=model.strip(), effort=effort or None)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


class LLMError(Exception):
    def __init__(self, message: str, *, provider: str, retryable: bool):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class LLMProvider(Protocol):
    name: str

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
    ) -> LLMResponse: ...
