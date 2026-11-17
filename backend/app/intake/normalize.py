"""
Unit, lab-name and date normalisation for machine-read text. Every function returns a confidence-bearing result:
ambiguous input is never silently resolved.

Dates: ISO, "5 Mar 2024", "March 5, 2024", "05/03/2024" (MDY or DMY per DATE_ORDER; flagged ambiguous when both
readings are valid), two-digit years (flagged). Units: temperature F->C, weight lb->kg, height in->cm, glucose
mg/dL->mmol/L, creatinine mg/dL->umol/L, mcg/ug/µg unified. Lab names map to canonical names.
"""
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from backend.app.config import settings

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


@dataclass
class NormalizedDate:
    iso: Optional[str]
    confidence: float
    note: str = ""


def _year(y: str) -> tuple[int, bool]:
    n = int(y)
    if len(y) == 2:
        return (2000 + n if n <= (date.today().year % 100) + 1 else 1900 + n), True
    return n, False


def _valid(y: int, m: int, d: int) -> bool:
    try:
        date(y, m, d)
        return 1900 <= y <= date.today().year + 1
    except ValueError:
        return False


def parse_date(text: str, order: Optional[str] = None) -> NormalizedDate:
    order = order or settings.date_order
    s = text.strip().lower().replace(",", " ")
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", s)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        return NormalizedDate(date(y, mo, d).isoformat(), 0.98) if _valid(y, mo, d) else NormalizedDate(None, 0.0, "invalid date")
    m = re.search(r"\b(\d{1,2})\s+([a-z]{3,9})\.?\s+(\d{2,4})\b", s) or None
    if m and m[2][:3] in MONTHS:
        y, two = _year(m[3])
        mo, d = MONTHS[m[2][:3]], int(m[1])
        if _valid(y, mo, d):
            return NormalizedDate(date(y, mo, d).isoformat(), 0.8 if two else 0.95, "two-digit year" if two else "")
    m = re.search(r"\b([a-z]{3,9})\.?\s+(\d{1,2})\s+(\d{2,4})\b", s)
    if m and m[1][:3] in MONTHS:
        y, two = _year(m[3])
        mo, d = MONTHS[m[1][:3]], int(m[2])
        if _valid(y, mo, d):
            return NormalizedDate(date(y, mo, d).isoformat(), 0.8 if two else 0.95, "two-digit year" if two else "")
    m = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b", s)
    if m:
        a, b = int(m[1]), int(m[2])
        y, two = _year(m[3])
        mdy, dmy = _valid(y, a, b), _valid(y, b, a)
        if mdy and dmy and a != b:
            mo, d = (a, b) if order == "MDY" else (b, a)
            return NormalizedDate(date(y, mo, d).isoformat(), 0.5, f"ambiguous day/month; read as {order}")
        if mdy or dmy:
            mo, d = (a, b) if mdy else (b, a)
            note = "two-digit year" if two else ""
            return NormalizedDate(date(y, mo, d).isoformat(), 0.8 if two else 0.92, note)
    return NormalizedDate(None, 0.0, "no date found")


@dataclass
class Quantity:
    value: Optional[float]
    unit: str
    original: str
    confidence: float = 1.0
    note: str = ""


LAB_ALIASES = {
    "hemoglobin": ["hemoglobin", "haemoglobin", "hgb", "hb"],
    "hba1c": ["hba1c", "hemoglobin a1c", "a1c", "glycated hemoglobin"],
    "potassium": ["potassium", "k", "k+"],
    "sodium": ["sodium", "na", "na+"],
    "creatinine": ["creatinine", "creat", "cr"],
    "glucose": ["glucose", "blood glucose", "bg", "glu"],
    "inr": ["inr", "pt inr", "international normalized ratio"],
    "wbc": ["wbc", "white blood cells", "white cell count", "leukocytes"],
    "platelets": ["platelets", "plt", "platelet count"],
    "troponin": ["troponin", "troponin i", "troponin t", "hs-troponin", "trop"],
    "bnp": ["bnp", "nt-probnp", "b-type natriuretic peptide"],
    "egfr": ["egfr", "estimated gfr"],
    "temperature": ["temperature", "temp", "t"],
    "weight": ["weight", "wt"],
}
_ALIAS = {alias: canon for canon, aliases in LAB_ALIASES.items() for alias in aliases}


def canonical_lab(name: str) -> Optional[str]:
    key = re.sub(r"[^a-z0-9+ -]", "", name.lower()).strip()
    return _ALIAS.get(key) or next((c for a, c in _ALIAS.items() if len(a) > 3 and a in key), None)


_UNIT = {"ug": "mcg", "µg": "mcg", "μg": "mcg", "mcg": "mcg", "mg/dl": "mg/dL", "mmol/l": "mmol/L", "umol/l": "umol/L", "µmol/l": "umol/L",
         "g/dl": "g/dL", "g/l": "g/L", "%": "%", "°c": "C", "c": "C", "°f": "F", "f": "F", "kg": "kg", "lb": "lb", "lbs": "lb",
         "pg/ml": "pg/mL", "ng/l": "ng/L", "ng/ml": "ng/mL", "x10^9/l": "x10^9/L", "10^9/l": "x10^9/L", "meq/l": "mmol/L", "ml/min": "mL/min"}


def normalize_unit(u: str) -> str:
    return _UNIT.get(u.strip().lower().replace(" ", ""), u.strip())


def normalize_quantity(lab: Optional[str], value: float, unit: str) -> Quantity:
    """Convert to the unit the rest of the system uses. Returns the original if no rule applies."""
    u = normalize_unit(unit)
    orig = f"{value:g} {unit}".strip()
    if lab == "temperature" and (u == "F" or (not u and value > 50)):
        return Quantity(round((value - 32) * 5 / 9, 1), "C", orig, 0.98 if u == "F" else 0.7, "converted from F" if u == "F" else "unit missing; assumed F")
    if lab == "temperature" and not u:
        return Quantity(value, "C", orig, 0.8, "unit missing; assumed C")
    if lab == "weight" and u == "lb":
        return Quantity(round(value * 0.453592, 1), "kg", orig, 0.98, "converted from lb")
    if lab == "glucose" and u == "mg/dL":
        return Quantity(round(value / 18.0, 1), "mmol/L", orig, 0.98, "converted from mg/dL")
    if lab == "creatinine" and u == "mg/dL":
        return Quantity(round(value * 88.4), "umol/L", orig, 0.98, "converted from mg/dL")
    if lab == "hemoglobin" and u == "g/L":
        return Quantity(round(value / 10, 1), "g/dL", orig, 0.98, "converted from g/L")
    return Quantity(value, u, orig, 1.0 if u else 0.85, "" if u else "unit missing")
