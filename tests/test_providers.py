from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest

from assistant.llm import LLMError, Message, ToolCall, ToolResult, ToolSpec
from assistant.llm.anthropic_provider import AnthropicProvider
from assistant.llm.openai_provider import OpenAIProvider

TOOLS = [ToolSpec("hr__get_leave_balance", "balance", {"type": "object", "properties": {}})]


class _Recorder:
    def __init__(self, response=None, error=None):
        self.calls = []
        self._response = response
        self._error = error

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


def _claude_response(**overrides):
    base = dict(
        content=[
            NS(type="thinking", thinking="", signature="sig"),
            NS(type="text", text="Checking."),
            NS(type="tool_use", id="toolu_1", name="hr__get_leave_balance", input={}),
        ],
        stop_reason="tool_use",
        stop_details=None,
        model="claude-opus-5-5",
        usage=NS(input_tokens=50, output_tokens=10, cache_read_input_tokens=40, cache_creation_input_tokens=0),
    )
    base.update(overrides)
    return NS(**base)


def _anthropic(response=None, error=None, fallbacks=True):
    stable, beta = _Recorder(response, error), _Recorder(response, error)
    client = NS(messages=stable, beta=NS(messages=beta))
    return AnthropicProvider(client, fallbacks=fallbacks), stable, beta


async def test_anthropic_request_shape_and_parse():
    provider, stable, beta = _anthropic(_claude_response())
    history = [
        Message("user", "hi"),
        Message("assistant", "hello"),
        Message("user", "balance?"),
    ]
    resp = await provider.complete(
        model="claude-opus-5-5", system="sys", messages=history, tools=TOOLS, max_tokens=1000, effort="high"
    )
    assert not stable.calls and len(beta.calls) == 1
    req = beta.calls[0]
    assert req["betas"] == ["server-side-fallback-2026-07-01"] and req["fallbacks"] == "default"
    assert req["output_config"] == {"effort": "high"}
    assert req["cache_control"] == {"type": "ephemeral"}
    for banned in ("temperature", "top_p", "top_k", "tool_choice", "thinking"):
        assert banned not in req
    assert req["messages"][1] == {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}

    assert resp.stop_reason == "tool_use" and resp.text == "Checking."
    assert resp.tool_calls[0].name == "hr__get_leave_balance"
    assert resp.usage.cache_read_tokens == 40
    assert resp.native[0].type == "thinking"  # kept for verbatim replay inside the turn


async def test_anthropic_haiku_uses_stable_endpoint_without_fallbacks():
    provider, stable, beta = _anthropic(_claude_response(model="claude-haiku-5-5"))
    await provider.complete(model="claude-haiku-5-5", system="s", messages=[Message("user", "x")])
    assert len(stable.calls) == 1 and not beta.calls
    assert "fallbacks" not in stable.calls[0] and "tools" not in stable.calls[0]


async def test_anthropic_tool_loop_replays_native_blocks_and_batches_results():
    first = _claude_response()
    provider, _, beta = _anthropic(_claude_response(content=[NS(type="text", text="18 days")], stop_reason="end_turn"))
    assistant_turn = Message(
        "assistant",
        "Checking.",
        [ToolCall("toolu_1", "hr__get_leave_balance", {}), ToolCall("toolu_2", "jira__get_my_tickets", {})],
        native=first.content,
        native_provider="anthropic",
    )
    results = Message("tool", tool_results=[ToolResult("toolu_1", '{"annual": 18}'), ToolResult("toolu_2", "boom", True)])
    await provider.complete(model="claude-opus-5-5", system="s", messages=[Message("user", "q"), assistant_turn, results])
    msgs = beta.calls[0]["messages"]
    assert msgs[1]["content"] is first.content
    assert msgs[2]["role"] == "user" and len(msgs[2]["content"]) == 2
    assert msgs[2]["content"][1] == {"type": "tool_result", "tool_use_id": "toolu_2", "content": "boom", "is_error": True}


async def test_anthropic_refusal_and_structured_output():
    refusal = _claude_response(
        content=[], stop_reason="refusal", stop_details=NS(category="cyber", explanation="")
    )
    provider, _, beta = _anthropic(refusal)
    resp = await provider.complete(
        model="claude-opus-5-5", system="s", messages=[Message("user", "x")], json_schema={"type": "object"}
    )
    assert resp.stop_reason == "refusal" and resp.refusal_category == "cyber"
    assert beta.calls[0]["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (529, True), (400, False)])
async def test_anthropic_errors_are_classified(status, retryable):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request, json={"error": {"type": "x", "message": "m"}})
    cls = {429: anthropic.RateLimitError, 529: anthropic.InternalServerError, 400: anthropic.BadRequestError}[status]
    provider, _, _ = _anthropic(error=cls("m", response=response, body=None))
    with pytest.raises(LLMError) as exc:
        await provider.complete(model="claude-opus-5-5", system="s", messages=[Message("user", "x")])
    assert exc.value.retryable is retryable


def _openai_response(message, finish="tool_calls"):
    return NS(
        choices=[NS(message=message, finish_reason=finish)],
        model="gpt-test",
        usage=NS(prompt_tokens=30, completion_tokens=5, prompt_tokens_details=NS(cached_tokens=10)),
    )


async def test_openai_request_shape_and_tool_parsing():
    message = NS(
        content=None,
        refusal=None,
        tool_calls=[
            NS(type="function", id="call_1", function=NS(name="hr__get_leave_balance", arguments="{}")),
            NS(type="function", id="call_2", function=NS(name="jira__get_ticket", arguments='{"key": ')),
        ],
    )
    rec = _Recorder(_openai_response(message))
    provider = OpenAIProvider(NS(chat=NS(completions=rec)), name="azure")
    history = [
        Message("user", "q"),
        Message("assistant", "", [ToolCall("call_0", "x", {"a": 1})]),
        Message("tool", tool_results=[ToolResult("call_0", "res"), ToolResult("call_9", "bad", True)]),
    ]
    resp = await provider.complete(model="dep", system="sys", messages=history, tools=TOOLS, effort="low")
    req = rec.calls[0]
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert req["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert req["messages"][3:] == [
        {"role": "tool", "tool_call_id": "call_0", "content": "res"},
        {"role": "tool", "tool_call_id": "call_9", "content": "ERROR: bad"},
    ]
    assert req["reasoning_effort"] == "low" and req["max_completion_tokens"] == 16000
    assert resp.stop_reason == "tool_use"
    assert [c.invalid_arguments for c in resp.tool_calls] == [False, True]
    assert resp.native["tool_calls"][1]["function"]["arguments"] == '{"key": '


async def test_native_payload_from_another_provider_is_not_replayed():
    rec = _Recorder(_openai_response(NS(content="ok", refusal=None, tool_calls=None), finish="stop"))
    provider = OpenAIProvider(NS(chat=NS(completions=rec)))
    foreign = Message("assistant", "hello", native=[NS(type="thinking")], native_provider="anthropic")
    resp = await provider.complete(model="m", system="s", messages=[Message("user", "q"), foreign])
    assert rec.calls[0]["messages"][2] == {"role": "assistant", "content": "hello"}
    assert resp.stop_reason == "end" and resp.usage.cache_read_tokens == 10
