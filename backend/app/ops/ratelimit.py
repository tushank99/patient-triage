"""
Rate limiting (fixed window per user, or per client IP when anonymous).

Buckets are chosen by route (triage, reassess, transcribe, upload, login, default) with limits from RATE_LIMITS.
Redis (REDIS_URL) makes limits shared across API instances; if Redis is missing or down the limiter degrades to a
per-process in-memory window instead of failing requests (an outage of the limiter must not become an outage of triage).
"""
import logging
import re
import threading
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.app.config import settings

logger = logging.getLogger(__name__)

RULES = [
    ("POST", re.compile(r"/triage$"), "triage"),
    ("POST", re.compile(r"/visits/[^/]+/reassess$"), "reassess"),
    ("POST", re.compile(r"/intake/transcribe$"), "transcribe"),
    ("POST", re.compile(r"/(patients/[^/]+/documents|documents/[^/]+/versions)$"), "upload"),
    ("POST", re.compile(r"/auth/login$"), "login"),
]
EXEMPT = re.compile(r"^/(healthz|readyz|metrics|docs|openapi.json|redoc)?$")


def bucket_for(method: str, path: str) -> str:
    for m, rx, name in RULES:
        if method == m and rx.search(path):
            return name
    return "default"


def parse_limit(spec: str) -> Tuple[int, int]:
    count, seconds = spec.split("/")
    return int(count), int(seconds)


class MemoryWindow:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))

    def hit(self, key: str, window: int) -> Tuple[int, int]:
        now = int(time.time())
        start = now - now % window
        with self._lock:
            w, n = self._counts[key]
            n = n + 1 if w == start else 1
            self._counts[key] = (start, n)
            if len(self._counts) > 50_000:  # bound memory
                for k in list(self._counts)[:10_000]:
                    self._counts.pop(k, None)
        return n, start + window - now


MEMORY = MemoryWindow()
_redis = None
_redis_failed_at = 0.0


def _get_redis():
    global _redis, _redis_failed_at
    if not settings.redis_url or time.time() - _redis_failed_at < 30:
        return None
    if _redis is None:
        import redis

        _redis = redis.Redis.from_url(settings.redis_url, socket_timeout=0.25, socket_connect_timeout=0.25)
    return _redis


def hit(key: str, window: int) -> Tuple[int, int, str]:
    """Returns (count in window, seconds until reset, backend used)."""
    global _redis_failed_at
    r = _get_redis()
    if r is not None:
        try:
            now = int(time.time())
            k = f"rl:{key}:{now - now % window}"
            pipe = r.pipeline()
            pipe.incr(k)
            pipe.expire(k, window + 1)
            n = pipe.execute()[0]
            return int(n), window - now % window, "redis"
        except Exception as e:  # degrade, do not fail the request
            _redis_failed_at = time.time()
            logger.warning("Rate limiter Redis unavailable (%s); using in-memory limits.", type(e).__name__)
    n, reset = MEMORY.hit(key, window)
    return n, reset, "memory"


def client_identity(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        try:  # verified decode: a forged token must not drain someone else's bucket
            sub = jwt.decode(auth[7:], settings.jwt_secret, algorithms=["HS256"]).get("sub")
            if sub:
                return f"user:{sub}"
        except jwt.PyJWTError:
            pass
    ip = request.client.host if request.client else "unknown"
    if settings.trust_proxy:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            ip = fwd.split(",")[0].strip()
    return f"ip:{ip}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not settings.rate_limit_enabled or EXEMPT.match(path) or request.method == "OPTIONS":
            return await call_next(request)
        bucket = bucket_for(request.method, path)
        limit, window = parse_limit(settings.rate_limits.get(bucket, settings.rate_limits["default"]))
        count, reset, _backend = hit(f"{bucket}:{client_identity(request)}", window)
        headers = {"X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": str(max(0, limit - count)), "X-RateLimit-Reset": str(reset)}
        if count > limit:
            from backend.app.ops.observability import RATE_LIMITED

            RATE_LIMITED.labels(bucket).inc()
            return JSONResponse(status_code=429, headers={**headers, "Retry-After": str(reset)},
                                content={"error": {"code": "rate_limited", "message": f"Too many {bucket} requests. Retry in {reset} s.", "details": None}})
        response = await call_next(request)
        for k, v in headers.items():
            response.headers[k] = v
        return response


def reset_for_tests() -> None:
    global _redis, _redis_failed_at
    MEMORY._counts.clear()
    _redis, _redis_failed_at = None, 0.0
