"""Any OpenAI-compatible endpoint: OpenAI, Groq, or a local server (Ollama, vLLM, llama.cpp)."""
import time
from typing import Optional
from urllib.parse import urlparse

from backend.app.llm.base import ProviderError, ProviderResponse

GROQ_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODELS = {"groq": "qwen/qwen3.8-27b", "openai": "gpt-4o-mini"}


def is_local_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "host.docker.internal") or host.endswith(".local") or host.startswith(("10.", "192.168."))


class OpenAICompatProvider:
    def __init__(self, api_key: str, model: str = "", base_url: str = ""):
        if api_key.startswith("gsk_") and not base_url:
            base_url, kind = GROQ_URL, "groq"
        else:
            kind = "local" if base_url and is_local_url(base_url) else ("custom" if base_url else "openai")
        self.name = kind
        self.base_url: Optional[str] = base_url or None
        self.model = model or DEFAULT_MODELS.get(kind, "gpt-4o-mini" if kind != "local" else "llama3.1")
        self.is_local = kind == "local"
        self._api_key = api_key or "local-no-key"
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            # max_retries=0: retrying is the assessor's job so attempts are counted and audited.
            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url, max_retries=0)
        return self._client

    def complete_json(self, system: str, user: str, timeout: float) -> ProviderResponse:
        import openai

        started = time.monotonic()
        try:
            resp = self._get_client().with_options(timeout=timeout).chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.1,
            )
        except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            raise ProviderError(type(e).__name__, transient=True) from e
        except openai.OpenAIError as e:
            raise ProviderError(type(e).__name__, transient=False) from e
        usage = getattr(resp, "usage", None)
        return ProviderResponse(
            text=resp.choices[0].message.content or "",
            latency_ms=int((time.monotonic() - started) * 1000),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
