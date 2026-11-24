"""
Single settings object for the whole backend. Everything secret comes from environment variables
(Render dashboard in deployment, backend/.env locally; .env is git-ignored). See docs/SECURITY.md.
"""
import logging
import os
import secrets
from typing import Dict, List, Literal, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger(__name__)

PROTECTED_ENVIRONMENTS = ("staging", "production")


class Settings(BaseSettings):
    environment: Literal["development", "test", "staging", "production"] = "development"
    facility_id: str = "ED-MAIN"

    database_url: str = "sqlite:///./triage.db"

    # ---- auth / secrets
    jwt_secret: str = ""  # REQUIRED (>= 32 chars) in staging/production; random per-process in dev/test
    jwt_expire_minutes: int = 30
    demo_user_password: str = ""  # password for the seeded demo accounts; see seed_users()
    login_max_failures: int = 5
    login_lockout_minutes: int = 15

    # ---- LLM providers (primary + optional fallback), both OpenAI-compatible
    openai_api_key: str = ""  # a gsk_... key routes to Groq
    llm_enabled: bool = True
    llm_model: str = ""
    llm_base_url: str = ""  # blank = provider default; set to http://localhost:11434/v1 for a local model
    llm_fallback_api_key: str = ""
    llm_fallback_model: str = ""
    llm_fallback_base_url: str = ""
    llm_timeout_seconds: float = 10.0
    llm_max_attempts: int = 2  # per provider, for transient errors and one JSON-repair retry
    llm_backoff_seconds: float = 0.5
    llm_circuit_failure_threshold: int = 3  # consecutive failed calls before a provider is skipped; 0 disables
    llm_circuit_cooldown_seconds: float = 30.0  # how long it is skipped before one trial call (half-open)
    # PHI gate: in staging/production a hosted (non-local) provider is blocked unless this is true,
    # which you may only set after signing a BAA with that provider. See docs/PHI_POLICY.md.
    llm_hosted_baa_confirmed: bool = False

    # ---- CORS
    cors_origins: List[str] = [
        "http://localhost:8080",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173",
    ]
    cors_origin_regex: Optional[str] = r"https://.*\.onrender\.com"

    # ---- document ingestion (Phase 3)
    max_upload_mb: float = 10.0
    max_pdf_pages: int = 300
    storage_backend: Literal["local", "s3"] = "local"
    storage_path: str = os.path.join(BACKEND_DIR, "data", "uploads")  # ephemeral on Render free: use s3 there
    s3_bucket: str = ""
    s3_endpoint_url: str = ""  # e.g. Cloudflare R2 / Backblaze B2 S3 endpoint; credentials via AWS_* env vars
    s3_prefix: str = "documents/"
    vector_backend: Literal["sql", "chroma"] = "sql"  # sql = vectors in the main DB (persistent on Neon)
    chroma_path: str = os.path.join(BACKEND_DIR, "data", "chroma")
    # Chosen by the Phase 4 benchmark (docs/RETRIEVAL_RESULTS.md); pinned in retrieval/models.lock.json.
    # "hash-v2" = offline lexical fallback (CI/tests).
    embedding_model: str = "fastembed:minishlab/potion-retrieval-32M"
    chunk_target_chars: int = 900
    chunk_overlap_chars: int = 150
    ingestion_worker_enabled: bool = True
    ingestion_max_attempts: int = 3
    ingestion_poll_seconds: float = 2.0
    ingestion_lease_minutes: int = 10
    clamav_host: str = ""  # set to scan uploads with clamd (INSTREAM); otherwise versions record scan_result=not_scanned
    clamav_port: int = 3310

    # ---- retrieval (Phase 4). See docs/RETRIEVAL.md for the benchmark behind the defaults.
    model_cache_dir: str = os.path.join(BACKEND_DIR, "data", "models")  # prefetch at BUILD time on Render
    allow_unpinned_models: bool = False
    reranker_model: str = ""  # off: the benchmark showed it LOWERS nDCG with the chosen embedder (+700 ms, +170 MB)
    retrieval_mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    retrieval_query_expansion: bool = True
    retrieval_parent_child: bool = True
    retrieval_candidates: int = 30
    retrieval_context_chars: int = 900

    # ---- multimodal intake (Phase 6). Models load lazily on first use. See docs/MULTIMODAL.md for RAM costs.
    stt_engine: Literal["faster-whisper", "none"] = "faster-whisper"
    stt_model: str = "tiny.en"  # WER benchmark winner with the medical vocabulary (docs/STT_RESULTS.md)
    stt_language: str = "en"  # ISO code, or "auto" to detect
    stt_compute_type: str = "int8"
    stt_min_word_confidence: float = 0.6
    max_audio_mb: float = 25.0
    max_audio_seconds: int = 300
    ocr_engine: Literal["rapidocr", "paddleocr", "none"] = "rapidocr"  # rapidocr = PaddleOCR (PP-OCR) models on ONNX Runtime
    ocr_min_line_confidence: float = 0.85
    ocr_min_field_confidence: float = 0.85
    date_order: Literal["MDY", "DMY"] = "MDY"  # how to read ambiguous dates like 03/04/2025 (always flagged)
    max_image_pixels: int = 40_000_000

    # ---- production hardening (Phase 7). See docs/PRODUCTION.md
    redis_url: str = ""  # e.g. Render Key Value / Upstash. Empty = in-process fallbacks (DB job queue, in-memory rate limits)
    job_backend: Literal["db", "celery"] = "db"  # celery = dispatch jobs to Celery workers over Redis (DB stays source of truth)
    rate_limit_enabled: bool = True
    rate_limits: Dict[str, str] = {  # "<count>/<seconds>" per user (or IP if anonymous)
        "triage": "30/60", "reassess": "60/60", "transcribe": "10/60", "upload": "20/60", "login": "20/60", "default": "300/60",
    }
    api_legacy_prefix: bool = True  # also serve /api/* (deprecated) next to /api/v1/*
    otel_enabled: bool = False  # OpenTelemetry tracing (OTEL_EXPORTER_OTLP_ENDPOINT etc. configure the exporter)
    metrics_token: str = ""  # bearer token for GET /metrics (Prometheus). Empty = metrics disabled in protected envs
    llm_prices_per_1k: Dict[str, List[float]] = {  # model -> [input USD, output USD] per 1k tokens, for cost metrics
        "qwen/qwen3.8-27b": [0.0003, 0.0006], "gpt-4o-mini": [0.00015, 0.0006], "llama-3.3-70b-versatile": [0.00059, 0.00079],
    }
    encryption_key: str = ""  # Fernet key: encrypts stored originals and backups. REQUIRED in staging/production
    backup_dir: str = os.path.join(BACKEND_DIR, "data", "backups")
    backup_keep: int = 14  # how many encrypted backups to keep (retention of backups == how long deleted data can linger)
    retention_access_log_days: int = 400
    retention_job_days: int = 30
    retention_capture_detail_days: int = 30  # word-level transcript data / OCR line boxes after verification
    retention_archive_anonymize_days: int = 0  # 0 = off. >0: anonymise patients whose last visit ended this long ago
    security_headers: bool = True
    trust_proxy: bool = True  # Render terminates TLS: client IP comes from X-Forwarded-For

    # ---- startup / demo
    seed_on_startup: bool = True
    seed_demo_users: bool = True
    seed_demo_queue: bool = True
    demo_mode: bool = True  # enables /api/demo/* (admin or charge nurse only)
    deterioration_monitor_enabled: bool = True
    monitor_speedup: float = 1.0  # >1 makes reassessment alerts fire sooner (demo)

    model_config = SettingsConfigDict(env_file=os.path.join(BACKEND_DIR, ".env"), extra="ignore")

    @field_validator("database_url")
    @classmethod
    def _normalise_postgres_scheme(cls, v: str) -> str:
        return "postgresql://" + v[len("postgres://"):] if v.startswith("postgres://") else v

    @field_validator("cors_origin_regex")
    @classmethod
    def _blank_regex_is_none(cls, v: Optional[str]) -> Optional[str]:
        return v or None

    @property
    def is_protected_env(self) -> bool:
        return self.environment in PROTECTED_ENVIRONMENTS

    def primary_llm(self) -> dict:
        return {"api_key": self.openai_api_key, "model": self.llm_model, "base_url": self.llm_base_url}

    def fallback_llm(self) -> dict:
        return {"api_key": self.llm_fallback_api_key, "model": self.llm_fallback_model, "base_url": self.llm_fallback_base_url}

    @property
    def llm_configured(self) -> bool:
        return self.llm_enabled and bool(self.openai_api_key or self.llm_base_url)


settings = Settings()


def ensure_secrets() -> None:
    """Called once at startup. Refuses to boot a protected environment with weak secrets or unsafe storage."""
    from backend.app.ops.security import production_checks

    problems = []
    jwt_ok = bool(settings.jwt_secret) and len(settings.jwt_secret) >= 32
    if settings.is_protected_env and not jwt_ok:
        problems.append("JWT_SECRET must be set to at least 32 random characters in staging/production.")
    problems += production_checks()
    if problems:
        raise RuntimeError("Refusing to start: " + " ".join(problems))
    if jwt_ok:
        return
    settings.jwt_secret = secrets.token_urlsafe(48)
    logger.warning("JWT_SECRET not set: using a random per-process secret (all sessions end on restart).")
