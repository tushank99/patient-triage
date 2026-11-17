"""
Structured clinical fields from OCR output, each with a confidence = OCR confidence of the source line(s) x the
reliability of the parse (unit conversion, date ambiguity). Fields below OCR_MIN_FIELD_CONFIDENCE are flagged.

  lab        canonical name, value, unit, normalised value/unit, reference range, H/L flag (tables and "Name: value unit" lines)
  medication drug name + dose (known drug list or "<Word> <number> mg")
  allergy    from an Allergies section or "allergic to X"
  date       document/collection dates, normalised to ISO (ambiguous day/month lowers confidence)
  identity   name / DOB / MRN printed on the page, so a mismatch with the selected patient can be caught before filing
"""
import re
from dataclasses import asdict, dataclass
from typing import List, Optional

from backend.app.clinical.facts import drug_data
from backend.app.config import settings
from backend.app.intake.normalize import canonical_lab, normalize_quantity, parse_date
from backend.app.intake.ocr import OcrResult

NUM = r"[<>]?\s*-?\d+(?:[.,]\d+)?"
LAB_LINE = re.compile(rf"^(?P<name>[A-Za-z][A-Za-z0-9 +()/-]{{0,30}}?)\s*[:\-]?\s+(?P<val>{NUM})\s*(?P<unit>[a-zA-Z%°µμ/^0-9.]*)\s*(?P<rest>.*)$")
RANGE = re.compile(rf"(?P<lo>{NUM})\s*[-–]\s*(?P<hi>{NUM})")
MED = re.compile(r"\b(?P<name>[A-Z][a-z]{3,}(?:-[a-z]+)?)\s+(?P<dose>\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?|ml)\b[^,;\n]{0,25})", re.I)
DATE_HINT = re.compile(r"\b(date|collected|dob|date of birth|admitted|discharged|reported|visit)\b", re.I)
DATE_ANY = re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{2,4}|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4})\b")
MRN = re.compile(r"\b(?:MRN|medical record (?:no|number))[:#\s]*([A-Z0-9-]{4,})", re.I)
NAME = re.compile(r"\b(?:patient(?: name)?|name)\s*[:\-]\s*([A-Z][A-Za-z'\-]+(?:,?\s+[A-Z][A-Za-z'\-]+){0,3})")


@dataclass
class OcrField:
    kind: str
    name: str
    value: Optional[str]
    unit: Optional[str] = None
    normalized_value: Optional[float] = None
    normalized_unit: Optional[str] = None
    reference_range: Optional[str] = None
    flag: Optional[str] = None
    source_text: str = ""
    page: Optional[int] = None
    confidence: float = 0.0
    low_confidence: bool = False
    note: str = ""


def _num(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ".").replace("<", "").replace(">", "").strip())
    except ValueError:
        return None


def _lab(name: str, val: str, unit: str, rest: str, src: str, page: int, conf: float) -> Optional[OcrField]:
    canon = canonical_lab(name)
    v = _num(val)
    if canon is None or v is None:
        return None
    q = normalize_quantity(canon, v, unit)
    rng = RANGE.search(rest or "")
    flag = "H" if re.search(r"\b(H|HIGH)\b", rest or "") else ("L" if re.search(r"\b(L|LOW)\b", rest or "") else None)
    if flag is None and rng and _num(rng["lo"]) is not None and _num(rng["hi"]) is not None:
        flag = "H" if v > _num(rng["hi"]) else ("L" if v < _num(rng["lo"]) else None)
    c = round(conf * q.confidence, 3)
    return OcrField("lab", canon, f"{v:g}", unit or None, q.value, q.unit, rng.group(0) if rng else None, flag, src, page, c,
                    c < settings.ocr_min_field_confidence, q.note)


def extract_fields(result: OcrResult) -> List[OcrField]:
    out: List[OcrField] = []
    drugs = set(drug_data()["anticoagulants"]) | {d for ds in drug_data()["allergy_classes"].values() for d in ds}
    thr = settings.ocr_min_field_confidence
    for page in result.pages:
        for t in page.tables:  # lab tables: first cell = test name
            for row, rc in zip(t.rows, t.row_confidence):
                if len(row) >= 2:
                    m = re.match(rf"^({NUM})\s*(.*)$", row[1])
                    if m:
                        unit = m[2] or (row[2] if len(row) > 2 and not re.search(r"\d", row[2]) else "")
                        rest = " ".join(row[2:])
                        f = _lab(row[0], m[1], unit, rest, " | ".join(row), page.page, rc)
                        if f:
                            out.append(f)
        section = ""
        for line in page.lines:
            if line.heading:
                section = line.text.lower()
            text = line.text
            m = LAB_LINE.match(text)
            if m and not any(f.kind == "lab" and f.source_text.startswith(text[:12]) for f in out):
                f = _lab(m["name"], m["val"], m["unit"], m["rest"], text, page.page, line.confidence)
                if f:
                    out.append(f)
            for mm in MED.finditer(text):
                if mm["name"].lower() in drugs or "medication" in section:
                    out.append(OcrField("medication", mm["name"].capitalize(), mm["dose"].strip(), source_text=text, page=page.page,
                                        confidence=line.confidence, low_confidence=line.confidence < thr))
            if "allerg" in section and not line.heading and text.strip():
                sub = re.split(r"[:\-(]", text, maxsplit=1)[0].strip()
                if sub and not re.match(r"(?i)no known|nkda|none", sub):
                    out.append(OcrField("allergy", sub, text, source_text=text, page=page.page, confidence=line.confidence,
                                        low_confidence=line.confidence < thr))
            for dm in DATE_ANY.finditer(text):
                nd = parse_date(dm.group(0))
                if nd.iso:
                    label = "dob" if re.search(r"(?i)dob|birth", text) else ("date" if DATE_HINT.search(text) else "date_mentioned")
                    c = round(line.confidence * nd.confidence, 3)
                    out.append(OcrField("date", label, nd.iso, source_text=text, page=page.page, confidence=c, low_confidence=c < thr, note=nd.note))
            mm = MRN.search(text)
            if mm:
                out.append(OcrField("identity", "mrn", mm[1], source_text=text, page=page.page, confidence=line.confidence, low_confidence=line.confidence < thr))
            nm = NAME.search(text)
            if nm:
                out.append(OcrField("identity", "name", nm[1].strip(), source_text=text, page=page.page, confidence=line.confidence,
                                    low_confidence=line.confidence < thr))
    return out


def fields_as_dicts(fields: List[OcrField]) -> List[dict]:
    return [asdict(f) for f in fields]
