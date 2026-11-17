"""
OCR for scanned documents and photos.

Engine: PaddleOCR's PP-OCR detection + recognition models. Default runtime is RapidOCR (the same models exported to
ONNX, run on onnxruntime: ~15 MB of models, no paddlepaddle install, fits a small instance). OCR_ENGINE=paddleocr uses
native PaddleOCR when it is installed. Both return line boxes with a confidence per line.

Preprocessing (OpenCV): grayscale -> upscale small scans -> denoise -> contrast (CLAHE) -> deskew (dominant text angle)
-> orientation (if confidence is poor, try 90/180/270 and keep the best).
Layout: lines sorted into reading order (columns detected by x-gaps), headings marked, rows/columns reconstructed into
tables when >= 3 aligned rows share column positions (lab tables).
Every line keeps its confidence; anything under OCR_MIN_LINE_CONFIDENCE is flagged for human review.
"""
import io
import logging
import statistics
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import List, Optional

import numpy as np

from backend.app.config import settings
from backend.app.ingestion.parsers import looks_like_heading

logger = logging.getLogger(__name__)


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: List[float]  # x0, y0, x1, y1 in the preprocessed page image
    page: int
    low: bool = False
    heading: bool = False


@dataclass
class OcrTable:
    page: int
    rows: List[List[str]]
    row_confidence: List[float]


@dataclass
class OcrPage:
    page: int
    lines: List[OcrLine]
    tables: List[OcrTable] = field(default_factory=list)
    rotation: int = 0
    skew_degrees: float = 0.0


@dataclass
class OcrResult:
    pages: List[OcrPage]
    engine: str

    @property
    def lines(self) -> List[OcrLine]:
        return [l for p in self.pages for l in p.lines]

    @property
    def confidence(self) -> Optional[float]:
        ls = self.lines
        return round(sum(l.confidence for l in ls) / len(ls), 3) if ls else None

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    def as_dict(self) -> dict:
        return {"engine": self.engine, "confidence": self.confidence, "pages": [asdict(p) for p in self.pages]}


# ------------------------------------------------------------------ preprocessing


def load_image(data: bytes) -> np.ndarray:
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels  # decompression-bomb guard
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return np.array(img)[:, :, ::-1]  # BGR for OpenCV


def deskew_angle(gray: np.ndarray) -> float:
    """Projection-profile deskew: rotate a downscaled binarised page over -6..+6 degrees and keep the angle whose
    horizontal ink profile is sharpest (text rows line up). Hough lines on Canny edges missed text skew entirely
    (OCR benchmark: 35% of auto-accepted fields wrong on 2.5-degree scans), because text does not form long lines."""
    import cv2

    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) if gray.shape[1] > 800 else gray
    binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if binary.mean() < 1:
        return 0.0
    h, w = binary.shape

    def sharpness(angle: float) -> float:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rows = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0).sum(axis=1).astype(np.float64)
        return float(np.var(rows))

    coarse = max(np.arange(-6.0, 6.01, 0.5), key=sharpness)
    fine = max(np.arange(coarse - 0.5, coarse + 0.51, 0.1), key=sharpness)
    return float(round(fine, 2))


def preprocess(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] < 1200:  # small photos/scans: upscale so glyphs are ~20-30 px tall
        f = 1200 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    angle = deskew_angle(gray)
    if abs(angle) > 0.3:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), angle


# ------------------------------------------------------------------ engines


class RapidOcrEngine:
    name = "rapidocr-ppocr-onnx"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        self._ocr = RapidOCR()

    def read(self, bgr: np.ndarray) -> List[tuple]:
        result, _ = self._ocr(bgr)
        out = []
        for box, text, conf in result or []:
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            out.append((text, float(conf), [min(xs), min(ys), max(xs), max(ys)]))
        return out


class PaddleOcrEngine:
    name = "paddleocr"

    def __init__(self):
        from paddleocr import PaddleOCR  # optional, heavy

        self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def read(self, bgr: np.ndarray) -> List[tuple]:
        out = []
        for page in self._ocr.ocr(bgr, cls=True) or []:
            for box, (text, conf) in page or []:
                xs, ys = [p[0] for p in box], [p[1] for p in box]
                out.append((text, float(conf), [min(xs), min(ys), max(xs), max(ys)]))
        return out


@lru_cache(maxsize=1)
def get_ocr_engine():
    if settings.ocr_engine == "none":
        return None
    try:
        return PaddleOcrEngine() if settings.ocr_engine == "paddleocr" else RapidOcrEngine()
    except ImportError:
        logger.warning("OCR engine %s is not installed (pip install -r backend/requirements-multimodal.txt); OCR disabled.", settings.ocr_engine)
        return None


# ------------------------------------------------------------------ layout & tables


def _reading_order(lines: List[OcrLine], width: float) -> List[OcrLine]:
    if not lines:
        return lines
    h = statistics.median([l.box[3] - l.box[1] for l in lines]) or 10
    # Two-column detection: a vertical gap free of text spanning most of the page height.
    lefts = sorted(l.box[0] for l in lines)
    gap_x = None
    for a, b in zip(lefts, lefts[1:]):
        if b - a > width * 0.25 and width * 0.3 < b < width * 0.7:
            gap_x = b
            break
    is_col2 = (lambda l: gap_x is not None and l.box[0] >= gap_x - 5 and all(not (o.box[0] < gap_x - 5 < o.box[2] and abs(o.box[1] - l.box[1]) < h) for o in lines))
    left = [l for l in lines if not is_col2(l)]
    right = [l for l in lines if is_col2(l)]
    key = lambda l: (round(l.box[1] / (h * 0.6)), l.box[0])  # noqa: E731
    return sorted(left, key=key) + sorted(right, key=key)


def reconstruct_tables(lines: List[OcrLine], page: int) -> List[OcrTable]:
    """Group lines into rows (similar y), keep rows with >= 2 cells, and accept a run of >= 3 consecutive such rows
    whose cells share column starts as a table."""
    if not lines:
        return []
    h = statistics.median([l.box[3] - l.box[1] for l in lines]) or 10
    rows: List[List[OcrLine]] = []
    for l in sorted(lines, key=lambda l: (l.box[1] + l.box[3]) / 2):
        mid = (l.box[1] + l.box[3]) / 2
        if rows and abs(mid - statistics.mean((r.box[1] + r.box[3]) / 2 for r in rows[-1])) < h * 0.5:
            rows[-1].append(l)
        else:
            rows.append([l])
    tables, run = [], []
    for r in rows + [[]]:
        cells = sorted(r, key=lambda l: l.box[0])
        if len(cells) >= 2 and (not run or abs(len(cells) - len(run[0])) <= 1):
            run.append(cells)
            continue
        if len(run) >= 3:
            tables.append(OcrTable(page, [[c.text for c in row] for row in run], [min(c.confidence for c in row) for row in run]))
        run = [cells] if len(cells) >= 2 else []
    return tables


def ocr_image(bgr: np.ndarray, page: int = 1, engine=None) -> OcrPage:
    engine = engine or get_ocr_engine()
    img, skew = preprocess(bgr)
    raw = engine.read(img)
    rotation = 0

    def vertical_share(res) -> float:  # sideways text shows up as boxes much taller than wide
        return (sum(1 for _, _, b in res if (b[3] - b[1]) > 1.5 * (b[2] - b[0])) / len(res)) if res else 0.0

    def title_on_top(res, height: int) -> bool:
        """The PP-OCR recogniser flips each upside-down crop on its own, so an upside-down page reads with good confidence;
        what reveals it is layout: the largest text (the title) ends up in the bottom half."""
        if not res:
            return False
        tallest = max(res, key=lambda r: r[2][3] - r[2][1])
        return (tallest[2][1] + tallest[2][3]) / 2 < height / 2

    def mean_conf(res) -> float:
        return (sum(c for _, c, _ in res) / len(res)) if res else 0.0

    if vertical_share(raw) > 0.4 or mean_conf(raw) < 0.8 or len(raw) < 3:
        import cv2

        candidates = [(0, img, raw)]
        for rot, code in ((90, cv2.ROTATE_90_CLOCKWISE), (270, cv2.ROTATE_90_COUNTERCLOCKWISE), (180, cv2.ROTATE_180)):
            cand = cv2.rotate(img, code)
            candidates.append((rot, cand, engine.read(cand)))
        rotation, img, raw = max(
            candidates,
            key=lambda c: (vertical_share(c[2]) <= 0.4, len(c[2]) >= 3, title_on_top(c[2], c[1].shape[0]), round(mean_conf(c[2]), 2)),
        )

    thr = settings.ocr_min_line_confidence
    lines = [OcrLine(t.strip(), round(c, 4), [float(v) for v in b], page, c < thr, looks_like_heading(t)) for t, c, b in raw if t.strip()]
    lines = _reading_order(lines, img.shape[1])
    return OcrPage(page, lines, reconstruct_tables(lines, page), rotation, round(skew, 2))


def ocr_document(data: bytes, mime_type: str, engine=None) -> OcrResult:
    engine = engine or get_ocr_engine()
    if engine is None:
        raise RuntimeError("OCR is disabled (OCR_ENGINE=none).")
    pages: List[OcrPage] = []
    if mime_type == "application/pdf":
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        try:
            for i, p in enumerate(doc, start=1):
                pix = p.get_pixmap(dpi=200)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                pages.append(ocr_image(arr[:, :, :3][:, :, ::-1].copy(), i, engine))
        finally:
            doc.close()
    else:
        pages.append(ocr_image(load_image(data), 1, engine))
    return OcrResult(pages, engine.name)
