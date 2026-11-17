"""
Metrics (Prometheus), tracing (OpenTelemetry) and latency/cost accounting.

  GET /metrics            Prometheus text format (bearer METRICS_TOKEN in staging/production)
  stage("retrieval")      context manager: times a pipeline stage into pipeline_stage_seconds and, when OTEL_ENABLED,
                          opens a span, so one trace shows API -> retrieval -> ML -> LLM -> rules -> grounding -> DB
  llm_cost(model, in, out) USD from LLM_PRICES_PER_1K; counted per model and stored on each assessment's audit entry
Queue depth and oldest-queued age are read from the database at scrape time (works for both job backends).
"""
import logging
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily, REGISTRY

from backend.app.config import settings

logger = logging.getLogger(__name__)

HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP latency by route template", ["method", "route", "status"],
                         buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30))
STAGE_LATENCY = Histogram("pipeline_stage_seconds", "Assessment pipeline stage latency", ["stage"],
                          buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30))
LLM_CALLS = Counter("llm_calls_total", "LLM calls", ["provider", "model", "outcome"])
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["model", "kind"])
LLM_COST = Counter("llm_cost_usd_total", "Estimated LLM spend (USD)", ["model"])
ASSESSMENTS = Counter("assessments_total", "Assessments produced", ["source", "degraded"])
DEGRADED = Counter("degraded_total", "Degraded pipeline stages", ["reason"])
JOBS_FINISHED = Counter("ingestion_jobs_finished_total", "Ingestion jobs finished", ["kind", "status"])
DOCS_INDEXED = Counter("documents_indexed_total", "Document versions indexed")
CHUNKS_INDEXED = Counter("chunks_indexed_total", "Chunks indexed")
RATE_LIMITED = Counter("rate_limited_total", "Requests rejected by the rate limiter", ["bucket"])
BUILD_INFO = Gauge("triage_build_info", "Build/model versions", ["pipeline", "rules", "embedding_model"])

_tracer = None


class QueueCollector:
    """Ingestion queue depth by status and age of the oldest due job, read at scrape time."""

    def collect(self):
        from sqlalchemy import func

        from backend.app.database import SessionLocal
        from backend.app.models import IngestionJob, utc_now, as_utc

        depth = GaugeMetricFamily("ingestion_jobs", "Ingestion jobs by status", labels=["status"])
        oldest = GaugeMetricFamily("ingestion_oldest_queued_seconds", "Age of the oldest QUEUED job")
        try:
            db = SessionLocal()
            try:
                for status, n in db.query(IngestionJob.status, func.count()).group_by(IngestionJob.status):
                    depth.add_metric([status], n)
                first = db.query(func.min(IngestionJob.created_at)).filter(IngestionJob.status == "QUEUED").scalar()
                oldest.add_metric([], (utc_now() - as_utc(first)).total_seconds() if first else 0.0)
            finally:
                db.close()
        except Exception:  # a scrape must never fail because the DB is briefly away
            return
        yield depth
        yield oldest


_registered = False


def setup_metrics() -> None:
    global _registered
    if not _registered:
        REGISTRY.register(QueueCollector())
        _registered = True


def setup_tracing(app) -> None:
    global _tracer
    if not settings.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "patient-triage-api", "deployment.environment": settings.environment}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # endpoint/headers from OTEL_EXPORTER_OTLP_* env
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")
        _tracer = trace.get_tracer("triage.pipeline")
        logger.info("OpenTelemetry tracing enabled")
    except Exception as e:
        logger.warning("Tracing not enabled: %s", e)


@contextmanager
def stage(name: str, **attributes) -> Iterator[None]:
    start = time.perf_counter()
    span_cm = _tracer.start_as_current_span(f"pipeline.{name}", attributes=attributes) if _tracer else None
    span = span_cm.__enter__() if span_cm else None
    try:
        yield
    except Exception as e:
        if span is not None:
            span.record_exception(e)
        raise
    finally:
        STAGE_LATENCY.labels(name).observe(time.perf_counter() - start)
        if span_cm:
            span_cm.__exit__(None, None, None)


def llm_cost(model: str, prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[float]:
    price = settings.llm_prices_per_1k.get(model)
    if price is None or prompt_tokens is None:
        return None
    return round((prompt_tokens or 0) / 1000 * price[0] + (completion_tokens or 0) / 1000 * price[1], 6)


def record_llm_call(provider: str, model: str, ok: bool, prompt_tokens, completion_tokens) -> Optional[float]:
    LLM_CALLS.labels(provider, model, "ok" if ok else "error").inc()
    if not ok:
        return None
    LLM_TOKENS.labels(model, "prompt").inc(prompt_tokens or 0)
    LLM_TOKENS.labels(model, "completion").inc(completion_tokens or 0)
    cost = llm_cost(model, prompt_tokens, completion_tokens)
    if cost:
        LLM_COST.labels(model).inc(cost)
    return cost


def metrics_response():
    from starlette.responses import Response

    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
