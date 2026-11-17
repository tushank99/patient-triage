"""
Parsers turn bytes into ordered Blocks (heading / paragraph / table row) with page numbers, preserving
reading order and section headings so the chunker can split on clinical sections.

  PDF   PyMuPDF (`pymupdf`). Headings from font size/bold + a clinical heading lexicon. NOTE: PyMuPDF is
        AGPL-3.0; a closed-source deployment needs Artifex's commercial licence or a swap to pypdf.
  DOCX  python-docx: paragraph styles ("Heading 1", "Title") and tables, in document order.
  TEXT  UTF-8 text: markdown '#', ALL-CAPS lines and "Heading:" lines are headings.
Scanned/image-only PDFs yield no text and fail with a clear message (OCR is Phase 5).
"""
import io
import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

from backend.app.ingestion.validation import DOCX, PDF, TEXT

PARSER_VERSION = "1"

CLINICAL_HEADINGS = {
    "chief complaint", "history of present illness", "hpi", "past medical history", "pmh", "past surgical history",
    "medications", "current medications", "medication list", "allergies", "social history", "family history",
    "review of systems", "ros", "physical exam", "physical examination", "vital signs", "vitals", "assessment",
    "plan", "assessment and plan", "impression", "diagnosis", "diagnoses", "discharge diagnosis", "discharge diagnoses",
    "hospital course", "discharge medications", "discharge instructions", "follow up", "follow-up", "labs",
    "laboratory results", "results", "imaging", "findings", "procedures", "problem list", "immunizations", "summary",
}


class ParseError(Exception):
    """Permanent: the document itself cannot be parsed (retrying will not help)."""


@dataclass
class Block:
    text: str
    kind: str  # heading | paragraph | table
    page: Optional[int] = None


@dataclass
class ParsedDocument:
    blocks: List[Block]
    parser: str
    page_count: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    bare = s.rstrip(":").strip().lower()
    if bare in CLINICAL_HEADINGS:
        return True
    if s.startswith("#"):
        return True
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters) and len(s.split()) <= 8 and not s.endswith("."):
        return True
    return bool(re.fullmatch(r"[A-Z][A-Za-z /&-]{2,50}:", s))


def _clean_heading(s: str) -> str:
    return s.strip().lstrip("#").strip().rstrip(":").strip()


def parse_text(data: bytes) -> ParsedDocument:
    text = data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    blocks: List[Block] = []
    for para in re.split(r"\n\s*\n", text):
        lines = [l for l in para.split("\n") if l.strip()]
        buf: List[str] = []
        for line in lines:
            if looks_like_heading(line):
                if buf:
                    blocks.append(Block(" ".join(buf).strip(), "paragraph"))
                    buf = []
                blocks.append(Block(_clean_heading(line), "heading"))
            else:
                buf.append(line.strip())
        if buf:
            blocks.append(Block(" ".join(buf).strip() if not all(l.lstrip().startswith(("-", "*", "•")) for l in buf) else "\n".join(buf), "paragraph"))
    return ParsedDocument(blocks, f"text-v{PARSER_VERSION}")


def parse_pdf(data: bytes) -> ParsedDocument:
    import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    try:
        lines = []  # (page, text, size, bold)
        for pno, page in enumerate(doc, start=1):
            for b in page.get_text("dict", sort=True)["blocks"]:
                for line in b.get("lines", []):
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    size = max(s["size"] for s in spans)
                    bold = all("bold" in s["font"].lower() or (s["flags"] & 16) for s in spans)
                    lines.append((pno, text, size, bold))
        page_count = doc.page_count
    finally:
        doc.close()
    if not lines:
        raise ParseError("No extractable text: the PDF looks scanned or image-only (OCR arrives in Phase 5).")
    body = statistics.median(l[2] for l in lines)
    blocks: List[Block] = []
    buf, buf_page = [], None
    for pno, text, size, bold in lines:
        heading = (size >= body * 1.15 and len(text) <= 80) or (bold and len(text) <= 60 and not text.endswith(".")) or looks_like_heading(text)
        if heading:
            if buf:
                blocks.append(Block(" ".join(buf), "paragraph", buf_page))
                buf = []
            blocks.append(Block(_clean_heading(text), "heading", pno))
            continue
        if buf and pno != buf_page:  # never let a paragraph straddle pages: keeps page provenance exact
            blocks.append(Block(" ".join(buf), "paragraph", buf_page))
            buf = []
        buf_page = pno if not buf else buf_page
        buf.append(text)
        if text.endswith((".", ":", "?", "!")) and len(" ".join(buf)) > 400:
            blocks.append(Block(" ".join(buf), "paragraph", buf_page))
            buf = []
    if buf:
        blocks.append(Block(" ".join(buf), "paragraph", buf_page))
    return ParsedDocument(blocks, f"pymupdf-v{PARSER_VERSION}", page_count)


def parse_docx(data: bytes) -> ParsedDocument:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(io.BytesIO(data))
    blocks: List[Block] = []
    for child in d.element.body.iterchildren():  # document order, paragraphs and tables interleaved
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "").lower()
            if style.startswith("heading") or style == "title" or looks_like_heading(text):
                blocks.append(Block(_clean_heading(text), "heading"))
            else:
                blocks.append(Block(text, "paragraph"))
        elif tag == "tbl":
            for row in Table(child, d).rows:
                cells = [c.text.strip() for c in row.cells]
                cells = [c for i, c in enumerate(cells) if c and (i == 0 or c != cells[i - 1])]  # merged cells repeat
                if cells:
                    blocks.append(Block(" | ".join(cells), "table"))
    if not blocks:
        raise ParseError("The Word document contains no text.")
    return ParsedDocument(blocks, f"python-docx-v{PARSER_VERSION}")


def parse(data: bytes, mime_type: str) -> ParsedDocument:
    try:
        if mime_type == PDF:
            return parse_pdf(data)
        if mime_type == DOCX:
            return parse_docx(data)
        if mime_type == TEXT:
            return parse_text(data)
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"Could not parse {mime_type}: {type(e).__name__}") from e
    raise ParseError(f"No parser for {mime_type}")
