"""Provider interface. The assessor depends only on this, never on a vendor SDK."""
from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class ProviderResponse:
    text: str
    latency_ms: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class ProviderError(Exception):
    """Raised by providers. `transient` errors (timeout, 429, 5xx, network) are retried; others are not."""

    def __init__(self, message: str, transient: bool):
        super().__init__(message)
        self.transient = transient


class LLMProvider(Protocol):
    name: str
    model: str
    is_local: bool

    def complete_json(self, system: str, user: str, timeout: float) -> ProviderResponse:
        """Return the model's raw text (a JSON object). Raise ProviderError on failure."""
        ...
