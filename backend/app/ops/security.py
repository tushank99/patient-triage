"""
Transport/at-rest protections and safe logging.

  SecurityHeadersMiddleware  nosniff, frame-deny, no-referrer, no-store for API responses, HSTS in staging/production
  PiiLogFilter               scrubs MRNs, phone numbers, e-mails and dates from every log record before it is written
  Fernet helpers             encrypt stored originals and backups (ENCRYPTION_KEY); EncryptedStorage wraps any storage
  production_checks()        refuses to start staging/production without TLS to the database or an encryption key
"""
import logging
import re

from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.config import settings

ENC_MAGIC = b"TRGENC1:"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if not settings.security_headers:
            return response
        h = response.headers
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "no-referrer")
        h.setdefault("Permissions-Policy", "camera=(), geolocation=(), microphone=()")
        if request.url.path.startswith("/api"):
            h.setdefault("Cache-Control", "no-store")  # PHI must not sit in shared/browser caches
            h.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        if settings.is_protected_env:
            h.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response


_PII = [
    (re.compile(r"\bMRN[-\s:#]*[A-Z0-9][A-Z0-9-]*\b", re.I), "[MRN]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"), "[PHONE]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
]


def scrub_text(text: str) -> str:
    for rx, token in _PII:
        text = rx.sub(token, text)
    return text


class PiiLogFilter(logging.Filter):
    """Scrubs the format string and each string argument separately, keeping `args` intact: formatters such as
    uvicorn's access log read record.args positionally, so collapsing them would break logging."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = scrub_text(record.msg)
            if isinstance(record.args, tuple):
                record.args = tuple(scrub_text(a) if isinstance(a, str) else a for a in record.args)
            elif isinstance(record.args, dict):
                record.args = {k: scrub_text(v) if isinstance(v, str) else v for k, v in record.args.items()}
        except Exception:
            pass
        return True


def install_log_filter() -> None:
    f = PiiLogFilter()
    root = logging.getLogger()
    for h in root.handlers or []:
        h.addFilter(f)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "backend"):
        logging.getLogger(name).addFilter(f)


def fernet():
    if not settings.encryption_key:
        return None
    from cryptography.fernet import Fernet

    return Fernet(settings.encryption_key.encode())


def encrypt(data: bytes) -> bytes:
    f = fernet()
    return ENC_MAGIC + f.encrypt(data) if f else data


def decrypt(data: bytes) -> bytes:
    if not data.startswith(ENC_MAGIC):
        return data  # written before encryption was enabled
    f = fernet()
    if f is None:
        raise RuntimeError("Encrypted object found but ENCRYPTION_KEY is not set.")
    return f.decrypt(data[len(ENC_MAGIC):])


class EncryptedStorage:
    """Wraps a Storage so every original document is encrypted with ENCRYPTION_KEY before it leaves the process."""

    def __init__(self, inner):
        self.inner, self.name = inner, f"{inner.name}+fernet"

    def put(self, key: str, data: bytes) -> None:
        self.inner.put(key, encrypt(data))

    def get(self, key: str) -> bytes:
        return decrypt(self.inner.get(key))

    def delete(self, key: str) -> None:
        self.inner.delete(key)

    def exists(self, key: str) -> bool:
        return self.inner.exists(key)


def production_checks() -> list:
    """Problems that must block a staging/production start (returned as messages)."""
    problems = []
    if not settings.is_protected_env:
        return problems
    url = settings.database_url
    if url.startswith("sqlite"):
        problems.append("SQLite is not a production database; set DATABASE_URL to Postgres.")
    elif "sslmode=require" not in url and "sslmode=verify" not in url:
        problems.append("DATABASE_URL must use TLS (?sslmode=require or verify-full).")
    if not settings.encryption_key:
        problems.append("ENCRYPTION_KEY is required (stored originals and backups are encrypted with it).")
    if settings.cors_origin_regex and ".*" in settings.cors_origin_regex:
        problems.append("CORS_ORIGIN_REGEX accepts any subdomain; set CORS_ORIGINS to your frontend URL and CORS_ORIGIN_REGEX=.")
    return problems
