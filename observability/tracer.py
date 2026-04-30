"""OTEL tracer for the Pi-GRPO API and trainers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import Settings


_provider: TracerProvider | None = None


def configure_tracing(settings: Settings) -> None:
    global _provider
    if _provider is not None:
        return
    res = Resource.create({"service.name": "pi-grpo", "service.version": settings.version})
    _provider = TracerProvider(resource=res)
    if settings.otel_endpoint:
        exporter = OTLPSpanExporter(endpoint=f"{settings.otel_endpoint}/v1/traces")
        _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    tracer = trace.get_tracer("pi.grpo")
    with tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else str(v))
        yield s
