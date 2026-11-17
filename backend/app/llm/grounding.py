"""
Citation validation: does the cited passage actually support the claim?

Every LLM claim must cite D-refs (record passages it was shown) or INTAKE (today's complaint + vitals). For each claim:
  invalid_citation  cites something it was never shown
  uncited           a claim about the record/history with no citation
  unsupported       cited text lacks the claim's key terms, a number differs, the negation is flipped, or the claim
                    says CURRENT where the source says resolved/historical/discontinued (or the reverse)
  partial           some but not enough key terms match
  supported         key terms (with abbreviation/synonym expansion of the source) and all numbers are present, no flips
Deterministic and explainable, so it can gate what the nurse sees. It is a lexical entailment check, not an NLI model:
it will miss a claim that copies the right words but reverses the meaning in a way these checks do not model. The
hallucination test set (backend/eval/hallucination_set.json) documents what it catches.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from backend.app.retrieval.normalize import STOPWORDS, normalize_query, tokenize

GENERIC = set(tokenize(
    "history historical documented document record records noted note notes patient reports reported previous prior has had "
    "with currently current resolved since known taking takes take on is was were the of and in for a an to past medical "
    "presents presenting today recently recent who which this that there their per also its at by as from which status record "
    "list listed shows showing indicates according mentions mention describes described evidence chart file"
))
NEG = re.compile(r"\b(no|not|denies|denied|without|negative|none|never|nkda|absent)\b", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")
INACTIVE = re.compile(r"\b(resolved|historical|discontinued|stopped|ceased|completed|former|previous|history of|delivered)\b", re.I)
ACTIVE = re.compile(r"\b(active|current|currently|ongoing|continued|daily|bid|tid|nightly)\b", re.I)

SUPPORT_MIN, PARTIAL_MIN = 0.7, 0.4


@dataclass
class ClaimCheck:
    text: str
    kind: str
    status: str
    citations: List[str]
    verdict: str  # supported | partial | unsupported | uncited | invalid_citation
    coverage: float = 0.0
    reasons: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)


def _source_tokens(text: str) -> set:
    toks = set(tokenize(text))
    toks |= set(tokenize(" ".join(normalize_query(text).expansion_terms)))  # "AF" in the source supports "atrial fibrillation"
    return toks


def _line_with(term_tokens: Sequence[str], text: str) -> str:
    """The source line/sentence that best matches the claim (for temporal and negation checks)."""
    best, best_n = "", 0
    for line in re.split(r"\n|(?<=[.;])\s", text):
        n = len(set(tokenize(line)) & set(term_tokens))
        if n > best_n:
            best, best_n = line, n
    return best


def check_claim(text: str, kind: str, status: str, citations: Sequence[str], sources: Dict[str, str],
                ref_to_doc: Optional[Dict[str, str]] = None) -> ClaimCheck:
    ref_to_doc = ref_to_doc or {}
    cites = [c.strip().upper() for c in citations if c and c.strip()]
    res = ClaimCheck(text, kind, status, cites, "supported", document_ids=[ref_to_doc[c] for c in cites if c in ref_to_doc])
    bad = [c for c in cites if c not in sources]
    if bad:
        res.verdict, res.reasons = "invalid_citation", [f"cites {', '.join(bad)}, which it was not shown"]
        return res
    if not cites:
        res.verdict = "uncited" if kind not in ("vitals", "complaint", "reasoning") else "supported"
        if res.verdict == "uncited":
            res.reasons = ["claim about the record has no citation"]
        return res

    source = "\n".join(sources[c] for c in cites)
    key = [t for t in dict.fromkeys(tokenize(text)) if t not in GENERIC and t not in STOPWORDS and not NUM.fullmatch(t)]
    have = _source_tokens(source)
    res.coverage = round(sum(t in have for t in key) / len(key), 3) if key else 1.0
    missing_nums = [n for n in NUM.findall(text) if n not in NUM.findall(source)]
    if missing_nums:
        res.reasons.append(f"number(s) {', '.join(missing_nums)} not in the cited text")
    line = _line_with(key, source)
    line = re.sub(r"\(?\bnot current\b\)?", "", line, flags=re.I)  # a status label, not a negated fact
    sentences = [s for s in re.split(r"\n|(?<=[.;])\s", source) if s.strip()]
    if NEG.search(text) and not any(NEG.search(s) and (set(tokenize(s)) & set(key) or not key) for s in sentences):
        res.reasons.append("claim is negative but the cited text is not")
    elif NEG.search(line) and not NEG.search(text) and kind not in ("vitals", "complaint"):
        res.reasons.append("cited text is negative but the claim is not")
    if status == "current" and INACTIVE.search(line) and not ACTIVE.search(line):
        res.reasons.append("claim says current; the cited text says resolved/historical/stopped")
    if status in ("resolved", "historical") and re.search(r"\bACTIVE\b", line) and not INACTIVE.search(line):
        res.reasons.append("claim says resolved; the cited text says active")

    if res.reasons:
        res.verdict = "unsupported"
    elif res.coverage >= SUPPORT_MIN:
        res.verdict = "supported"
    elif res.coverage >= PARTIAL_MIN:
        res.verdict, res.reasons = "partial", [f"only {res.coverage:.0%} of the claim's key terms appear in the cited text"]
    else:
        res.verdict, res.reasons = "unsupported", [f"only {res.coverage:.0%} of the claim's key terms appear in the cited text"]
    return res


@dataclass
class GroundingReport:
    checks: List[ClaimCheck]

    @property
    def counts(self) -> Dict[str, int]:
        out = {k: 0 for k in ("supported", "partial", "unsupported", "uncited", "invalid_citation")}
        for c in self.checks:
            out[c.verdict] += 1
        return out

    @property
    def score(self) -> Optional[float]:
        """Share of claims fully supported (None when the model made no claims)."""
        return round(self.counts["supported"] / len(self.checks), 3) if self.checks else None

    @property
    def problems(self) -> List[ClaimCheck]:
        return [c for c in self.checks if c.verdict in ("unsupported", "uncited", "invalid_citation")]


def validate_claims(claims, sources: Dict[str, str], ref_to_doc: Dict[str, str]) -> GroundingReport:
    return GroundingReport([check_claim(c.text, c.kind, c.status, c.citations, sources, ref_to_doc) for c in claims])
