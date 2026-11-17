"""
Query normalisation and expansion for clinical retrieval.

  normalize_query("fell at home, on a blood thinner, hx of afib")
    original   the text as typed
    variants   [original, original with abbreviations expanded, clinical-concepts-only]   (multi-query)
    expansion  the added clinical terms (BM25 weights them lower than the user's own words)

Shared tokenizer for BM25 and query/document matching lives here too, so both sides tokenize identically.
"""
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Tuple

LEXICON_PATH = os.path.join(os.path.dirname(__file__), "medical_lexicon.json")

STOPWORDS = frozenset(
    "a an and are as at be been by for from had has have he her his i in is it its me my no not of on or our she "
    "that the their them they this to was were will with without you your patient pt reports states since also "
    "very some any per day days today yesterday".split()
)
_TOKEN = re.compile(r"[a-z][a-z0-9/-]*[a-z0-9]|[a-z]|\d+(?:\.\d+)?")


@lru_cache(maxsize=1)
def lexicon() -> Dict:
    with open(LEXICON_PATH, encoding="utf-8") as f:
        return json.load(f)


def _stem(t: str) -> str:
    """Tiny plural folding; identical on both sides, so 'seizures' matches 'seizure'."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokenize(text: str) -> List[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class NormalizedQuery:
    original: str
    variants: List[str]
    expansion_terms: List[str] = field(default_factory=list)
    matched: List[Tuple[str, List[str]]] = field(default_factory=list)  # (surface form, expansions)

    @property
    def expanded_text(self) -> str:
        return self.original + (" " + " ".join(self.expansion_terms) if self.expansion_terms else "")

    def weighted_tokens(self, expansion_weight: float = 0.6) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for t in tokenize(self.original):
            weights[t] = 1.0
        for t in tokenize(" ".join(self.expansion_terms)):
            weights.setdefault(t, expansion_weight)
        return weights


def normalize_query(text: str, expand: bool = True) -> NormalizedQuery:
    original = " ".join(text.split())
    if not expand:
        return NormalizedQuery(original, [original])
    lex = lexicon()
    low = original.lower()
    matched: List[Tuple[str, List[str]]] = []
    # Lay phrases first, longest first, so "high blood pressure" wins over "blood pressure".
    for phrase in sorted(lex["lay_terms"], key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", low):
            matched.append((phrase, lex["lay_terms"][phrase]))
    rewritten = original
    for abbr, expansions in lex["abbreviations"].items():
        pattern = r"(?<![A-Za-z0-9])" + re.escape(abbr) + r"(?![A-Za-z0-9])"
        if re.search(pattern, original, flags=re.IGNORECASE):
            matched.append((abbr, expansions))
            rewritten = re.sub(pattern, f"{abbr} ({expansions[0]})", rewritten, flags=re.IGNORECASE)
    terms: List[str] = []
    for _, exps in matched:
        for e in exps:
            if e.lower() not in low and e not in terms:
                terms.append(e)
    variants = [original]
    if rewritten != original:
        variants.append(rewritten)
    if terms:
        variants.append(", ".join(terms))
    return NormalizedQuery(original, variants, terms, matched)
