"""OpenTelemetry setup. Spans emitted per turn:

    POST /v1/chat                      (FastAPI instrumentation)
    └── assistant.turn                 tier, model, cache hit, blocked
        ├── guardrails.input           heuristic score, PII kinds
        ├── chat <classifier model>    input classifier
        ├── rag.retrieve               doc ids, similarities
        ├── cache.lookup
        ├── chat <model>               GenAI semconv: tokens, finish reason
        └── tool <name>                rbac decision, confirmation, status
            └── tools/call <name>      (MCP SDK, client side; server side in the MCP process)
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_tracing(service_name: str, environment: str) -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, "deployment.environment": environment})
    )
    # Endpoint comes from OTEL_EXPORTER_OTLP_ENDPOINT (e.g. http://jaeger:4318).
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None
