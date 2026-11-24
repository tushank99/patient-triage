"""
Triage LLM stage: versioned prompt, validated structured output, retry, JSON repair, provider
fallback and a PHI gate. Provider-agnostic: it only sees the LLMProvider interface.
"""
import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.app.config import settings
from backend.app.llm.base import LLMProvider, ProviderError
from backend.app.llm.openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)

PROMPT_NAME = "triage_assessment.v4"
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

_sleep: Callable[[float], None] = time.sleep  # patched in tests
_now: Callable[[], float] = time.monotonic  # patched in tests


# ---- circuit breaker (docs/runbook.md, "LLM provider outage"). Without it, every triage during an outage paid
# the full retry budget again (measured: 143 ms -> 5,008 ms per request). After N consecutive failed calls a
# provider is skipped for a cooldown; the deterministic rules + ML result is returned at once, marked degraded.
# After the cooldown one call is let through (half-open): success closes the circuit, failure re-opens it.
@dataclass
class _Circuit:
    failures: int = 0
    open_until: float = 0.0


_CIRCUITS: dict = {}


def _circuit_key(provider: LLMProvider) -> str:
    return f"{provider.name}:{provider.model}"


def _circuit_open(provider: LLMProvider) -> bool:
    c = _CIRCUITS.get(_circuit_key(provider))
    return bool(c and settings.llm_circuit_failure_threshold > 0 and _now() < c.open_until)


def _circuit_result(provider: LLMProvider, ok: bool) -> None:
    key = _circuit_key(provider)
    c = _CIRCUITS.setdefault(key, _Circuit())
    if ok:
        c.failures, c.open_until = 0, 0.0
        return
    c.failures += 1
    threshold = settings.llm_circuit_failure_threshold
    if threshold > 0 and c.failures >= threshold:
        c.open_until = _now() + settings.llm_circuit_cooldown_seconds
        logger.error("LLM circuit OPEN for %s after %d consecutive failures; skipping it for %.0fs.",
                     key, c.failures, settings.llm_circuit_cooldown_seconds)


def reset_circuits() -> None:
    _CIRCUITS.clear()


class LLMClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., min_length=3, max_length=300)
    kind: Literal["condition", "medication", "allergy", "history", "contradiction", "vitals", "complaint", "reasoning"] = "history"
    status: Literal["current", "resolved", "historical", "unknown"] = "unknown"
    citations: List[str] = Field(default_factory=list, max_length=5)


class LLMAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: Literal["CARDIAC", "RESPIRATORY", "NEURO", "TRAUMA_LOCALIZED", "TRAUMA_MAJOR", "INFECTION", "OTHER"] = "OTHER"
    esi: int = Field(..., ge=1, le=5)
    rationale: str = Field(..., min_length=1, max_length=600)
    risk_factors: List[str] = Field(default_factory=list, max_length=3)
    cited_document_ids: List[str] = Field(default_factory=list, max_length=10)
    claims: List[LLMClaim] = Field(default_factory=list, max_length=12)


@dataclass
class LLMInput:
    """Everything here is already de-identified by the pipeline (names, MRN, contact details, dates)."""

    age_label: str  # exact years, or "90+" (ages over 89 are aggregated)
    gender: str
    vitals: dict
    complaint: str
    history_block: str  # documents labelled D1, D2, ... (never real document ids)
    facts_block: str = ""  # deterministic extracted facts and conflicts (also untrusted data)


@dataclass
class LLMCallMeta:
    provider: str
    model: str
    prompt_version: str
    attempts: int
    latency_ms: int
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    used_fallback: bool
    request_sha256: str
    request_text: str  # exactly what left the server (de-identified); stored in the audit trail
    cost_usd: Optional[float] = None  # estimate from LLM_PRICES_PER_1K


@dataclass
class LLMOutcome:
    assessment: LLMAssessment
    meta: LLMCallMeta


class LLMUnavailable(Exception):
    """`reason` is a short machine-readable code surfaced in the response's `degraded` list."""

    def __init__(self, reason: str, attempts: Optional[List[str]] = None):
        super().__init__(reason)
        self.reason = reason
        self.attempts = attempts or []


@lru_cache(maxsize=4)
def load_prompt(name: str = PROMPT_NAME) -> str:
    with open(os.path.join(PROMPT_DIR, f"{name}.md"), encoding="utf-8") as f:
        return f.read()


_TAG_BREAK = re.compile(r"</?\s*(history|complaint|facts)\s*>", re.IGNORECASE)


def _neutralise(text: str) -> str:
    """Stop untrusted text from closing/opening our delimiter tags."""
    return _TAG_BREAK.sub("[tag removed]", text)


def build_user_prompt(inp: LLMInput) -> str:
    v = inp.vitals

    def fmt(x, unit=""):
        return "not measured" if x is None else f"{x:g}{unit}" if isinstance(x, (int, float)) else str(x)

    bp = "not measured" if v.get("systolic_bp") is None else f"{v['systolic_bp']}/{fmt(v.get('diastolic_bp'))}"
    return (
        f"Age: {inp.age_label} | Sex: {inp.gender}\n"
        f"Vitals: HR {fmt(v.get('heart_rate'))}, BP {bp}, RR {fmt(v.get('respiratory_rate'))}, "
        f"SpO2 {fmt(v.get('oxygen_saturation'), '%')}, Temp {fmt(v.get('temperature_celsius'), 'C')}, "
        f"Pain {fmt(v.get('pain_score'), '/10')}\n"
        f"<complaint>\n{_neutralise(inp.complaint)}\n</complaint>\n"
        f"<facts>\n{_neutralise(inp.facts_block or 'none')}\n</facts>\n"
        f"<history>\n{_neutralise(inp.history_block)}\n</history>"
    )


def build_provider_chain(cfg=settings) -> List[LLMProvider]:
    """Primary then fallback, minus any provider the PHI policy forbids."""
    chain: List[LLMProvider] = []
    if not cfg.llm_enabled:
        return chain
    for conf in (cfg.primary_llm(), cfg.fallback_llm()):
        if not (conf["api_key"] or conf["base_url"]):
            continue
        provider = OpenAICompatProvider(conf["api_key"], conf["model"], conf["base_url"])
        if cfg.is_protected_env and not provider.is_local and not cfg.llm_hosted_baa_confirmed:
            logger.warning("PHI policy: hosted LLM provider %r blocked in %s (no BAA confirmed).", provider.name, cfg.environment)
            continue
        chain.append(provider)
    return chain


# ---- observability: in-process ring buffer of recent calls (exposed at GET /api/ai/metrics)
CALLS: deque = deque(maxlen=500)


def _record(provider: str, model: str, ok: bool, reason: str, latency_ms: int, attempts: int, prompt_tokens, completion_tokens) -> Optional[float]:
    from backend.app.ops.observability import record_llm_call

    cost = record_llm_call(provider, model, ok, prompt_tokens, completion_tokens)
    CALLS.append({"at": time.time(), "provider": provider, "model": model, "ok": ok, "reason": reason, "latency_ms": latency_ms,
                  "attempts": attempts, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "prompt": PROMPT_NAME, "cost_usd": cost})
    return cost


def metrics_summary() -> dict:
    calls = list(CALLS)
    by_model: dict = {}
    for c in calls:
        m = by_model.setdefault(f"{c['provider']}:{c['model']}", {"calls": 0, "failures": 0, "latency": [], "prompt_tokens": 0, "completion_tokens": 0})
        m["calls"] += 1
        m["failures"] += 0 if c["ok"] else 1
        if c["ok"]:
            m["latency"].append(c["latency_ms"])
        m["prompt_tokens"] += c["prompt_tokens"] or 0
        m["completion_tokens"] += c["completion_tokens"] or 0
    out = {}
    for k, m in by_model.items():
        lat = sorted(m.pop("latency"))
        m["latency_p50_ms"] = lat[len(lat) // 2] if lat else None
        m["latency_p95_ms"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None
        out[k] = m
    reasons: dict = {}
    for c in calls:
        if not c["ok"]:
            reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
    circuits = {k: {"open": _now() < c.open_until, "consecutive_failures": c.failures} for k, c in _CIRCUITS.items()}
    return {"window_calls": len(calls), "prompt_version": PROMPT_NAME, "by_model": out, "failure_reasons": reasons, "circuits": circuits}


def _parse(text: str) -> LLMAssessment:
    return LLMAssessment.model_validate(json.loads(text))


def assess_with_llm(inp: LLMInput, chain: Optional[List[LLMProvider]] = None) -> LLMOutcome:
    """Raises LLMUnavailable. Per provider: up to llm_max_attempts tries; transient errors back off,
    invalid JSON gets one repair prompt; permanent errors move straight to the next provider."""
    chain = build_provider_chain() if chain is None else chain
    if not chain:
        reason = "phi_policy_blocked" if settings.is_protected_env and settings.llm_configured else "llm_not_configured"
        raise LLMUnavailable(reason)

    system = load_prompt()
    user = build_user_prompt(inp)
    request_text = f"[system: {PROMPT_NAME}]\n{user}"
    request_sha = hashlib.sha256((system + user).encode()).hexdigest()
    log: List[str] = []
    attempts = 0

    for idx, provider in enumerate(chain):
        if _circuit_open(provider):
            log.append(f"{provider.name}:circuit_open")
            continue
        prompt = user
        provider_error = False
        for attempt in range(1, settings.llm_max_attempts + 1):
            attempts += 1
            try:
                resp = provider.complete_json(system, prompt, settings.llm_timeout_seconds)
                assessment = _parse(resp.text)
                _circuit_result(provider, ok=True)
                cost = _record(provider.name, provider.model, True, "", resp.latency_ms, attempts, resp.prompt_tokens, resp.completion_tokens)
                return LLMOutcome(assessment, LLMCallMeta(cost_usd=cost, 
                    provider=provider.name, model=provider.model, prompt_version=PROMPT_NAME, attempts=attempts,
                    latency_ms=resp.latency_ms, prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                    used_fallback=idx > 0, request_sha256=request_sha, request_text=request_text,
                ))
            except ProviderError as e:
                provider_error = True
                log.append(f"{provider.name}:{e}")
                if not e.transient:
                    break  # permanent (auth, bad request): try the next provider
                if attempt < settings.llm_max_attempts:
                    _sleep(settings.llm_backoff_seconds * (2 ** (attempt - 1)))
            except (ValidationError, json.JSONDecodeError) as e:
                log.append(f"{provider.name}:invalid_output")
                # One repair attempt: tell the model what was wrong. Never echo the model's own text back.
                prompt = user + "\n\nYour previous reply was not valid JSON matching the required schema. Reply with ONLY the JSON object."
                logger.warning("LLM invalid output (%s)", type(e).__name__)
        if provider_error:  # unreachable/erroring provider; bad JSON from a reachable one does not trip the breaker
            _circuit_result(provider, ok=False)
    if all(entry.endswith(":circuit_open") for entry in log):  # every provider skipped: nothing was called
        logger.warning("LLM skipped: circuit open for every provider %s", log)
        raise LLMUnavailable("llm_unavailable", log)
    _record(chain[-1].name, chain[-1].model, False, log[-1] if log else "unknown", 0, attempts, None, None)
    logger.warning("LLM unavailable after %d attempt(s): %s", attempts, log)
    raise LLMUnavailable("llm_unavailable", log)
