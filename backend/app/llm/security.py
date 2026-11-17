"""
Prompt-injection defence for UNTRUSTED text (retrieved documents, uploaded files, free-text intake).

Layers:
  1. Delimiters: untrusted text sits inside <history>/<complaint>/<facts>; tag look-alikes are neutralised.
  2. Detection + neutralisation (this module): sentences that read like instructions to an AI ("ignore previous
     instructions", "system:", "set the ESI to 5", "you are now ...") are replaced by a marker before the prompt is
     built, and the request is flagged for review and audit.
  3. The system prompt states that delimited text is data, never instructions.
  4. Output side: structured output only; the safety-rule floor cannot be lowered by anything the model says; every
     history claim must cite a passage and is verified against it (grounding.py).
Regexes cannot catch every paraphrase, which is why layers 3-4 exist and why no single layer is trusted.
"""
import re
from dataclasses import dataclass, field
from typing import List

PATTERNS = [
    r"\bignore\b.{0,40}\b(instructions?|rules|prompt|above|previous|prior)\b",
    r"\bdisregard\b.{0,40}\b(instructions?|rules|prompt|above|previous|prior)\b",
    r"\b(forget|override|bypass)\b.{0,30}\b(instructions?|rules|filters?|guardrails?|safety)\b",
    r"(^|\n|\.\s)\s*(system|assistant|developer|user)\s*:",
    r"\byou are (now|actually)\b",
    r"\b(set|change|lower|assign|output|return)\b.{0,20}\b(esi|acuity|triage level)\b.{0,15}\b(to|=|of)\b\s*[1-5]",
    r"\b(respond|reply|answer|output)\b.{0,20}\b(only )?with\b.{0,30}\b(json|esi|\{)",
    r"\bnew instructions?\b",
    r"\bdo not (mention|cite|flag|report|escalate)\b",
    r"\bjailbreak\b|\bprompt injection\b",
    r"\binclude (all|every) patients?'?s? records\b",
    r"</?\s*(system|history|complaint|facts|instructions?)\s*>",
]
_RX = re.compile("|".join(f"(?:{p})" for p in PATTERNS), re.I)
_SENT = re.compile(r"[^.!?\n]*[.!?\n]?")
MARKER = "[instruction-like text withheld]"


@dataclass
class InjectionScan:
    text: str
    hits: List[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.hits)


def neutralise(text: str) -> InjectionScan:
    """Replaces every sentence containing an instruction-like pattern with MARKER."""
    if not text or not _RX.search(text):
        return InjectionScan(text)
    hits: List[str] = []
    out: List[str] = []
    for sent in _SENT.findall(text):
        if sent and _RX.search(sent):
            hits.append(sent.strip()[:120])
            out.append(MARKER + (" " if sent.endswith((" ", ".")) else ""))
        else:
            out.append(sent)
    return InjectionScan("".join(out), hits)
