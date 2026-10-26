"""HTTP API.

    POST /v1/chat                         ask a question or request an action
    POST /v1/actions/{id}/confirm|cancel  approve or reject a held action
    GET  /v1/conversations/{id}           your own conversation history
    GET  /v1/me                           identity, roles and the tools you can use
    GET  /v1/admin/prompt-logs            prompt log (admin:prompt_logs:read)
    GET  /healthz, /readyz
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from assistant.auth.rbac import Principal
from assistant.auth.tokens import TokenError, principal_from_claims
from assistant.config import get_settings
from assistant.factory import Runtime, build_runtime
from assistant.llm import LLMError
from assistant.observability.tracing import setup_tracing
from assistant.orchestrator import ActionForbidden, ActionNotFound
from assistant.sessions import ConversationNotFound

log = logging.getLogger("assistant.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = await build_runtime(get_settings())
    app.state.runtime = runtime
    try:
        yield
    finally:
        await runtime.close()


settings = get_settings()
app = FastAPI(title="Company Assistant", version="0.1.0", lifespan=lifespan)
if settings.tracing_enabled:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    setup_tracing(settings.otel_service_name, settings.environment)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz")

bearer = HTTPBearer(auto_error=False)


@app.exception_handler(LLMError)
async def llm_unavailable(request: Request, exc: LLMError) -> JSONResponse:
    log.error("LLM provider %s failed: %s", exc.provider, exc)
    return JSONResponse(
        {"detail": "The assistant's language model is unavailable right now. Please try again shortly."},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


def runtime(request: Request) -> Runtime:
    return request.app.state.runtime


async def current_principal(
    rt: Annotated[Runtime, Depends(runtime)],
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED, "invalid or missing bearer token", headers={"WWW-Authenticate": "Bearer"}
    )
    if creds is None:
        raise unauthorized
    try:
        claims = await run_in_threadpool(rt.validator.validate, creds.credentials)
    except TokenError as e:
        log.info("rejected token: %s", e)
        raise unauthorized from e
    principal = principal_from_claims(claims, rt.rbac, settings.oidc_roles_claim, token=creds.credentials)
    if not principal.roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "your account has no assistant role")
    return principal


def require(permission: str):
    async def check(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
        if not principal.has(permission):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"missing permission {permission}")
        return principal

    return check


PrincipalDep = Annotated[Principal, Depends(current_principal)]
RuntimeDep = Annotated[Runtime, Depends(runtime)]


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = None
    force_tier: Literal["small", "frontier"] | None = Field(
        default=None, description="Override the router (needs admin:router:override)"
    )


class ChatOut(BaseModel):
    conversation_id: str
    answer: str
    citations: list[dict[str, Any]]
    pending_actions: list[dict[str, Any]]
    route: dict[str, Any] | None
    cache_hit: bool
    blocked: bool
    trace_id: str | None


@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, principal: PrincipalDep, rt: RuntimeDep) -> ChatOut:
    if body.force_tier and not principal.has("admin:router:override"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "missing permission admin:router:override")
    if not await rt.sessions.allow_request(principal.sub, settings.rate_limit_per_minute):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded, try again in a minute")
    try:
        result = await rt.assistant.chat(
            principal, body.message, conversation_id=body.conversation_id, force_tier=body.force_tier
        )
    except ConversationNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found") from e
    return ChatOut(
        conversation_id=result.conversation_id,
        answer=result.answer,
        citations=[vars(c) for c in result.citations],
        pending_actions=[vars(a) for a in result.pending_actions],
        route=result.route,
        cache_hit=result.cache_hit,
        blocked=result.blocked,
        trace_id=result.trace_id,
    )


@app.post("/v1/actions/{action_id}/{decision}")
async def decide_action(action_id: str, decision: Literal["confirm", "cancel"], principal: PrincipalDep, rt: RuntimeDep):
    try:
        if decision == "confirm":
            outcome = await rt.assistant.confirm_action(principal, action_id)
        else:
            outcome = await rt.assistant.cancel_action(principal, action_id)
    except ActionNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "action not found or expired") from e
    except ActionForbidden as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "you are no longer permitted to run this action") from e
    return vars(outcome)


@app.get("/v1/conversations/{conversation_id}")
async def conversation(conversation_id: str, principal: PrincipalDep, rt: RuntimeDep):
    conv = await rt.sessions.get(conversation_id, principal.sub)
    if conv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    return {
        "conversation_id": conv.id,
        "messages": [{"role": role, "content": conv.vault.restore(text)} for role, text in conv.history],
    }


@app.get("/v1/me")
async def me(principal: PrincipalDep, rt: RuntimeDep):
    tools = await rt.assistant.tools.list_tools(principal)
    return {
        "sub": principal.sub,
        "username": principal.username,
        "roles": sorted(principal.roles),
        "permissions": sorted(principal.permissions),
        "tools": [{"name": t.name, "requires_confirmation": t.policy.confirm} for t in tools],
    }


@app.get("/v1/admin/prompt-logs")
async def prompt_logs(
    principal: Annotated[Principal, Depends(require("admin:prompt_logs:read"))],
    rt: RuntimeDep,
    limit: int = 50,
    user_sub: str | None = None,
):
    return await rt.prompt_log.recent(min(max(limit, 1), 500), user_sub)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(rt: RuntimeDep):
    try:
        await rt.pool.fetchval("SELECT 1")
        await rt.redis.ping()
    except Exception as e:
        return JSONResponse({"status": "unavailable", "error": type(e).__name__}, status_code=503)
    return {"status": "ready"}
