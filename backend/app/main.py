import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import InterfaceError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.background_tasks import deterioration_monitor
from backend.app.ingestion.worker import ingestion_worker_loop, safety_net_loop
from backend.app.config import ensure_secrets, settings
from backend.app.db_migrate import run_migrations
from backend.app.deps import get_llm_fn  # noqa: F401  (re-exported for tests)
from backend.app.ops.observability import HTTP_LATENCY, metrics_response, setup_metrics, setup_tracing
from backend.app.ops.ratelimit import RateLimitMiddleware
from backend.app.ops.security import SecurityHeadersMiddleware, install_log_filter
from backend.app.routers import auth, demo, documents, intake, queue, triage
from backend.app.schemas import ErrorResponse

API_V1 = "/api/v1"
LEGACY_SUNSET = "Wed, 30 Jun 2027 00:00:00 GMT"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    install_log_filter()  # scrub MRNs/phones/e-mails/dates from every log line
    setup_metrics()
    ensure_secrets()  # refuses to start staging/production with a weak JWT secret
    run_migrations()  # Alembic; stamps pre-Alembic databases at the baseline first
    if settings.seed_on_startup:
        try:
            from backend.scripts.seed_data import seed

            logger.info("Seed: %s", seed(users=settings.seed_demo_users, queue=settings.seed_demo_queue))
        except Exception as e:  # never block startup on seeding
            logger.error("Seeding failed: %s", e)
    tasks = []
    if settings.job_backend == "db":  # single-process mode (Render free tier): loops run inside the API
        if settings.deterioration_monitor_enabled:
            tasks.append(asyncio.create_task(deterioration_monitor()))
        if settings.ingestion_worker_enabled:
            tasks.append(asyncio.create_task(ingestion_worker_loop()))
    elif settings.ingestion_worker_enabled:  # Celery workers + beat do the work; this only rescues lost messages
        tasks.append(asyncio.create_task(safety_net_loop()))
    yield
    for task in tasks:
        if task:
            task.cancel()


app = FastAPI(
    title="PatientTriage.ai",
    description="AI-assisted ED triage prototype. Synthetic data only; not for clinical use. "
                "Versioned API under /api/v1; the unversioned /api/* aliases are deprecated.",
    version="0.4.0",
    lifespan=lifespan,
    responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 404: {"model": ErrorResponse},
               409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=False,  # bearer tokens travel in the Authorization header, never cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware order (outermost last): latency -> security headers -> rate limit -> CORS -> routes.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


@app.middleware("http")
async def latency_and_deprecation(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    template = getattr(route, "path", None) or "unmatched"  # templates, not raw paths: bounded label cardinality
    HTTP_LATENCY.labels(request.method, template, str(response.status_code)).observe(time.perf_counter() - t0)
    path = request.url.path
    if path.startswith("/api/") and not path.startswith(API_V1 + "/"):
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = LEGACY_SUNSET
        response.headers["Link"] = f'<{API_V1}{path[4:]}>; rel="successor-version"'
    return response


ROUTERS = (auth.router, triage.router, queue.router, demo.router, documents.router, intake.router)
v1 = APIRouter(prefix=API_V1)
for r in ROUTERS:
    v1.include_router(r)
app.include_router(v1)
if settings.api_legacy_prefix:  # old clients keep working; hidden from the schema so new code targets /api/v1
    legacy = APIRouter(prefix="/api", include_in_schema=False)
    for r in ROUTERS:
        legacy.include_router(r)
    app.include_router(legacy)
setup_tracing(app)

# ---------------------------------------------------------------- error envelope
# Every error has the same shape: {"error": {"code", "message", "details"}}.

_STATUS_CODES = {400: "bad_request", 401: "unauthenticated", 403: "forbidden", 404: "not_found", 409: "conflict", 413: "payload_too_large", 415: "unsupported_media_type", 503: "unavailable", 422: "validation_error", 429: "rate_limited"}


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException):
    code = _STATUS_CODES.get(exc.status_code, "error")
    return JSONResponse(status_code=exc.status_code, headers=getattr(exc, "headers", None),
                        content={"error": {"code": code, "message": str(exc.detail), "details": None}})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": "Request does not match the API contract.", "details": details}})


@app.exception_handler(OperationalError)
@app.exception_handler(InterfaceError)
async def database_error_handler(_: Request, exc: Exception):
    # Database unreachable/locked: tell the client to retry instead of a bare 500. The frontend's edge
    # fallback (ONNX + rules in the browser) takes over for triage while this lasts.
    logger.error("Database error: %s", type(exc).__name__)
    return JSONResponse(status_code=503, headers={"Retry-After": "5"},
                        content={"error": {"code": "database_unavailable", "message": "Database temporarily unavailable; retry shortly.", "details": None}})


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "Internal server error.", "details": None}})


@app.get("/", include_in_schema=False)
async def health_check():
    return {"status": "healthy", "message": "PatientTriage.ai Backend is running."}


@app.get("/healthz", include_in_schema=False)
async def liveness():
    """Liveness: the process answers. Never touches dependencies (a DB outage must not get the pod killed)."""
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
async def readiness():
    """Readiness: can this instance serve triage? DB is required; Redis/ML degrade but do not fail readiness."""
    from backend.app.database import engine
    from backend.app.ml.model import get_model

    checks, ok = {}, True
    try:
        def _ping():
            with engine.connect() as c:
                c.execute(text("SELECT 1"))
        await asyncio.wait_for(asyncio.to_thread(_ping), timeout=3)
        checks["database"] = "ok"
    except Exception as e:
        checks["database"], ok = f"error: {type(e).__name__}", False
    checks["ml_model"] = "ok" if get_model() is not None else "degraded (rules + LLM only)"
    if settings.redis_url:
        from backend.app.ops.ratelimit import _get_redis

        try:
            r = _get_redis()
            await asyncio.wait_for(asyncio.to_thread(r.ping), timeout=1)
            checks["redis"] = "ok"
        except Exception:
            checks["redis"] = "degraded (in-memory rate limits; jobs via DB safety net)"
    checks["job_backend"] = settings.job_backend
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ready" if ok else "not_ready", "checks": checks})


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    """Prometheus scrape. In staging/production it requires `Authorization: Bearer $METRICS_TOKEN`."""
    if settings.is_protected_env or settings.metrics_token:
        import hmac

        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not settings.metrics_token or not hmac.compare_digest(given, settings.metrics_token):
            return JSONResponse(status_code=401, content={"error": {"code": "unauthenticated", "message": "metrics token required", "details": None}})
    return metrics_response()
