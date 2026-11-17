"""
OCR benchmark: synthetic lab reports and medication lists rendered to images, degraded, OCR'd, and scored.

  python -m backend.eval.ocr_benchmark --report docs/OCR_RESULTS.md

Metrics per degradation: character error rate (CER), field-level accuracy (labs: normalised value + unit; dates;
medications; allergies; MRN), and the SAFETY metric for the review gate: of the fields that were auto-accepted
(confidence >= threshold), how many were wrong ("auto-accept error rate"), and of the wrong fields, how many the
gate held for review ("error catch rate").
Rendered text is cleaner than real scans and phone photos: treat these as relative numbers.
"""
import argparse
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings  # noqa: E402
from backend.app.intake.ocr import OcrResult, get_ocr_engine, ocr_image  # noqa: E402
from backend.app.intake.ocr_extract import extract_fields  # noqa: E402

LABS = [("Hemoglobin", "g/dL", (9.0, 16.5), "12.0-16.0", "hemoglobin", 1.0), ("Potassium", "mmol/L", (3.0, 6.2), "3.5-5.1", "potassium", 1.0),
        ("Sodium", "mmol/L", (128, 148), "135-145", "sodium", 1.0), ("Creatinine", "mg/dL", (0.6, 3.5), "0.6-1.2", "creatinine", 88.4),
        ("Glucose", "mg/dL", (70, 320), "70-99", "glucose", 1 / 18), ("INR", "", (0.9, 4.5), "0.8-1.2", "inr", 1.0),
        ("WBC", "x10^9/L", (3.0, 18.0), "4.0-11.0", "wbc", 1.0), ("HbA1c", "%", (5.0, 11.0), "4.0-5.6", "hba1c", 1.0)]
MEDS = ["Apixaban 5 mg twice daily", "Warfarin 4 mg nightly", "Metoprolol 50 mg daily", "Lisinopril 20 mg daily", "Metformin 1000 mg twice daily",
        "Atorvastatin 40 mg nightly", "Furosemide 40 mg daily", "Levothyroxine 75 mcg daily"]
ALLERGIES = ["Penicillin - hives", "Sulfa - rash", "Latex - urticaria", "Codeine - confusion"]
NAMES = ["Morgan, Alex", "Rivera, Sam", "Okafor, Ngozi", "Chen, Wei", "Patel, Ravi", "Novak, Ana"]
DATE_FORMATS = ["%Y-%m-%d", "%d %b %Y", "%B %d, %Y", "%m/%d/%Y"]


def find_font(size: int):
    from PIL import ImageFont

    for path in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_document(i: int, rng: random.Random):
    from datetime import date, timedelta

    d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 600))
    fmt = DATE_FORMATS[i % len(DATE_FORMATS)]
    date_text = d.strftime(fmt)
    if fmt == "%m/%d/%Y" and d.day <= 12:  # keep the benchmark unambiguous; ambiguity is unit-tested separately
        d = d.replace(day=13 + (d.day % 15))
        date_text = d.strftime(fmt)
    mrn = f"MRN-{rng.randint(1000, 9999)}"
    lines, truth = [], {"labs": {}, "date": d.isoformat(), "mrn": mrn, "meds": [], "allergies": []}
    table_rows = []
    if i % 2 == 0:
        lines += [("CITY HOSPITAL LABORATORY REPORT", 34, 0), (f"Patient: {rng.choice(NAMES)}", 26, 0), (f"MRN: {mrn}", 26, 0),
                  (f"Collected: {date_text}", 26, 0), ("", 20, 0)]
        table_rows.append(("Test", "Result", "Units", "Reference"))
        for name, unit, (lo, hi), ref, canon, factor in rng.sample(LABS, 5):
            v = round(rng.uniform(lo, hi), 1)
            table_rows.append((name, f"{v:g}", unit, ref))
            truth["labs"][canon] = (round(v * factor, 1) if factor != 1.0 else v)
    else:
        meds, allergies = rng.sample(MEDS, 4), rng.sample(ALLERGIES, 2)
        lines += [("CLINIC MEDICATION RECONCILIATION", 34, 0), (f"MRN: {mrn}", 26, 0), (f"Visit date: {date_text}", 26, 0), ("", 20, 0),
                  ("MEDICATIONS", 30, 0)] + [(m, 26, 40) for m in meds] + [("", 20, 0), ("ALLERGIES", 30, 0)] + [(a, 26, 40) for a in allergies]
        truth["meds"] = [m.split()[0].lower() for m in meds]
        truth["allergies"] = [a.split(" - ")[0].lower() for a in allergies]
    return lines, table_rows, truth


def render(lines, table_rows):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(img)
    y = 120
    gt_text = []
    for text, size, indent in lines:
        if text:
            draw.text((120 + indent, y), text, fill="black", font=find_font(size))
            gt_text.append(text)
        y += int(size * 1.9)
    cols = [120, 620, 920, 1220]
    for r, row in enumerate(table_rows):
        for x, cell in zip(cols, row):
            if cell:
                draw.text((x, y), cell, fill="black", font=find_font(30 if r == 0 else 28))
        gt_text.append(" ".join(c for c in row if c))
        y += 62
    return img, "\n".join(gt_text)


def degrade(img, kind: str, rng: random.Random):
    from PIL import Image, ImageEnhance, ImageFilter

    if kind == "clean":
        return img
    if kind == "skew 2.5deg":
        return img.rotate(2.5, expand=True, fillcolor="white")
    if kind == "noise+blur":
        arr = np.array(img).astype(np.int16)
        arr += np.random.default_rng(rng.randint(0, 9999)).normal(0, 28, arr.shape).astype(np.int16)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.4))
    if kind == "low contrast":
        return ImageEnhance.Contrast(img).enhance(0.25)
    if kind == "jpeg q15 + downscale":
        small = img.resize((img.width // 2, img.height // 2))
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=15)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind == "rotated 90deg":
        return img.rotate(90, expand=True, fillcolor="white")
    raise ValueError(kind)


def cer(ref: str, hyp: str) -> float:
    r, h = " ".join(ref.split()), " ".join(hyp.split())
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def score(fields, truth):
    """Returns [(correct: bool, auto_accepted: bool)] for every ground-truth field (missing counts as wrong, not accepted)."""
    out = []
    for canon, val in truth["labs"].items():
        f = next((f for f in fields if f.kind == "lab" and f.name == canon), None)
        ok = f is not None and f.normalized_value is not None and abs(f.normalized_value - val) <= max(0.051, abs(val) * 0.01)
        out.append(("lab", ok, f is not None and not f.low_confidence))
    f = next((f for f in fields if f.kind == "date" and f.name == "date"), None)
    out.append(("date", f is not None and f.value == truth["date"], f is not None and not f.low_confidence))
    f = next((f for f in fields if f.kind == "identity" and f.name == "mrn"), None)
    out.append(("mrn", f is not None and f.value.upper() == truth["mrn"], f is not None and not f.low_confidence))
    for m in truth["meds"]:
        f = next((f for f in fields if f.kind == "medication" and f.name.lower() == m), None)
        out.append(("medication", f is not None, f is not None and not f.low_confidence))
    for a in truth["allergies"]:
        f = next((f for f in fields if f.kind == "allergy" and a in f.name.lower()), None)
        out.append(("allergy", f is not None, f is not None and not f.low_confidence))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "backend" / "eval" / "ocr_results.json"))
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    engine = get_ocr_engine()
    rng = random.Random(11)
    docs = [make_document(i, rng) for i in range(a.docs)]
    results = {}
    for kind in ["clean", "skew 2.5deg", "noise+blur", "low contrast", "jpeg q15 + downscale", "rotated 90deg"]:
        cers, scored, conf = [], [], []
        for lines, rows, truth in docs:
            img, gt = render(lines, rows)
            page = ocr_image(np.array(degrade(img, kind, rng))[:, :, ::-1].copy(), 1, engine)
            res = OcrResult([page], engine.name)
            cers.append(cer(gt, res.text))
            scored += score(extract_fields(res), truth)
            conf.append(res.confidence or 0)
        correct = [s for s in scored if s[1]]
        accepted = [s for s in scored if s[2]]
        wrong = [s for s in scored if not s[1]]
        results[kind] = {
            "cer": float(np.mean(cers)), "field_accuracy": len(correct) / len(scored), "n_fields": len(scored),
            "by_kind": {k: sum(1 for s in scored if s[0] == k and s[1]) / max(1, sum(1 for s in scored if s[0] == k)) for k in ("lab", "date", "mrn", "medication", "allergy")},
            "auto_accept_rate": len(accepted) / len(scored),
            "auto_accept_error_rate": (sum(1 for s in accepted if not s[1]) / len(accepted)) if accepted else None,
            "error_catch_rate": (sum(1 for s in wrong if not s[2]) / len(wrong)) if wrong else None,
            "mean_line_confidence": float(np.mean(conf)),
        }
        print(f"{kind:<22} CER {results[kind]['cer']:.3f}  field acc {results[kind]['field_accuracy']:.3f}  auto-accepted {results[kind]['auto_accept_rate']:.2f}  "
              f"wrong-but-accepted {results[kind]['auto_accept_error_rate']}  errors held {results[kind]['error_catch_rate']}")
    Path(a.out).write_text(json.dumps({"engine": engine.name, "line_threshold": settings.ocr_min_line_confidence,
                                       "field_threshold": settings.ocr_min_field_confidence, "results": results}, indent=2), encoding="utf-8")
    if a.report:
        f = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
        lines = [f"# OCR benchmark ({engine.name})", "", f"{a.docs} synthetic documents (lab reports with tables, medication/allergy lists) x 6 degradations. "
                 f"Field threshold {settings.ocr_min_field_confidence}.", "",
                 "| degradation | CER | field accuracy | labs | dates | MRN | meds | allergies | auto-accepted | wrong among auto-accepted | errors held for review |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in results.items():
            b = r["by_kind"]
            lines.append(f"| {k} | {r['cer']:.3f} | {r['field_accuracy']:.2f} | {f(b['lab'])} | {f(b['date'])} | {f(b['mrn'])} | {f(b['medication'])} | {f(b['allergy'])} | "
                         f"{r['auto_accept_rate']:.2f} | {f(r['auto_accept_error_rate'])} | {f(r['error_catch_rate'])} |")
        Path(a.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
