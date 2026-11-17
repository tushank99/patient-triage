"""
The one PII scrubber. Replaces middleware/pii_scrubber.py and services/pii_service.py.

Runs on every piece of text that leaves the server (complaint + retrieved history sent to
the LLM). Regex-based, so it is a floor, not a guarantee: Phase 1+ should evaluate Presidio.
All patterns use word boundaries (the old name list redacted "Doe" inside "does").
Fail-closed: if scrubbing throws, the text is withheld entirely.
"""
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Eponyms that follow an honorific-like pattern but are disease names, e.g. "Dr. Parkinson's protocol".
MEDICAL_EPONYMS = {"parkinson", "alzheimer", "hodgkin", "crohn", "down", "graves", "addison", "cushing", "bell"}

_PATTERNS = [
    ("[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("[PHONE]", re.compile(r"(?<![\d/])(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![\d/])")),
    ("[MRN]", re.compile(r"\bMRN[\s:#-]*[A-Z0-9][A-Z0-9-]*\b", re.IGNORECASE)),
    ("[DATE]", re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
]
_HONORIFIC = re.compile(r"\b(Mr|Mrs|Ms|Miss|Dr)\.?\s+([A-Z][a-zA-Z'-]+)")


@dataclass
class ScrubResult:
    text: str
    redactions: int


def scrub_pii(text: Optional[str], known_names: Iterable[str] = ()) -> ScrubResult:
    if not text:
        return ScrubResult(text or "", 0)
    try:
        count = 0
        out = text

        # 1. Structured identifiers first, so name replacement cannot break an email/MRN pattern.
        for token, pattern in _PATTERNS:
            out, n = pattern.subn(token, out)
            count += n

        # 2. Names we know belong to this patient ("Doe, John" -> "Doe", "John").
        parts = {p for name in known_names if name for p in re.split(r"[\s,]+", name) if len(p) >= 2}
        for part in sorted(parts, key=len, reverse=True):
            out, n = re.subn(r"\b" + re.escape(part) + r"\b", "[NAME]", out, flags=re.IGNORECASE)
            count += n

        # 3. Honorific + capitalised word, unless it is a disease eponym.
        def _honorific(m: re.Match) -> str:
            nonlocal count
            if re.sub(r"'s$", "", m.group(2).lower()) in MEDICAL_EPONYMS:
                return m.group(0)
            count += 1
            return f"{m.group(1)} [NAME]"

        out = _HONORIFIC.sub(_honorific, out)
        return ScrubResult(out, count)
    except Exception as e:  # pragma: no cover - defensive
        logger.error("PII scrubber failed (%s). Withholding text.", type(e).__name__)
        return ScrubResult("[TEXT WITHHELD: PII SCRUBBER FAILED]", 1)


_ISO_DATE = re.compile(r"(?<![\d-])(\d{4})-(\d{2})(?:-(\d{2}))?(?![\d-])")


def relative_age_label(d: Optional[date], today: date) -> str:
    """'3 months ago' / '2.5 years ago'. Keeps temporal reasoning without sending real dates."""
    if d is None:
        return "undated"
    days = (today - d).days
    if days < 0:
        return "in the future"
    if days < 45:
        return "within the last 6 weeks"
    if days < 700:
        return f"{round(days / 30.4)} months ago"
    return f"{days / 365.25:.1f} years ago"


def generalize_dates(text: str, today: date) -> str:
    """Replace ISO dates (YYYY-MM[-DD]) with relative phrases BEFORE scrubbing, so the LLM can still tell
    'resolved 2021' from 'last month' but never sees a real calendar date."""

    def _sub(m: re.Match) -> str:
        try:
            return relative_age_label(date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)), today)
        except ValueError:
            return m.group(0)

    return _ISO_DATE.sub(_sub, text)
