"""
Upload validation. Runs synchronously at upload time, before anything is stored, and never trusts the
client's filename, extension or Content-Type: the type is decided from the bytes.

Accepted: PDF, DOCX, plain text (UTF-8). Rejected with a reason: oversize, unknown type, encrypted PDFs,
PDFs with active content (JavaScript, launch actions, embedded files), macro-enabled or oversized/zip-bomb
DOCX, binary data posing as text, and anything the malware scanner flags.
"""
import io
import re
import socket
import struct
import zipfile
from dataclasses import dataclass
from typing import Optional

from backend.app.config import settings

PDF, DOCX, TEXT = "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"
PNG, JPEG, TIFF = "image/png", "image/jpeg", "image/tiff"
IMAGES = (PNG, JPEG, TIFF)

MAX_DOCX_UNCOMPRESSED = 50 * 1024 * 1024
MAX_DOCX_ENTRIES = 2000
MAX_ZIP_RATIO = 100
_PDF_ACTIVE = re.compile(rb"/(JavaScript|JS|Launch|EmbeddedFile|RichMedia|OpenAction\s*<<[^>]*?/S\s*/JavaScript)\b")


class UploadRejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class ValidatedUpload:
    data: bytes
    mime_type: str
    scan_result: str  # clean | not_scanned


def max_bytes() -> int:
    return int(settings.max_upload_mb * 1024 * 1024)


def sniff(data: bytes) -> Optional[str]:
    if data.startswith(b"%PDF-"):
        return PDF
    if data.startswith(b"PK\x03\x04"):
        return DOCX  # confirmed (or rejected) by _check_docx
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return PNG
    if data.startswith(b"\xff\xd8\xff"):
        return JPEG
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return TIFF
    return TEXT if _looks_like_text(data) else None


def _looks_like_text(data: bytes) -> bool:
    if b"\x00" in data[:8192]:
        return False
    try:
        sample = data[:65536].decode("utf-8")
    except UnicodeDecodeError as e:
        if e.start < len(data[:65536]) - 4:  # a multi-byte char cut at the sample edge is fine
            return False
        sample = data[: e.start].decode("utf-8")
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in sample)
    return bool(sample) and printable / len(sample) > 0.95


def _check_pdf(data: bytes) -> None:
    if _PDF_ACTIVE.search(data):
        raise UploadRejected(415, "PDF contains active content (JavaScript, launch actions or embedded files). Upload a flattened/printed PDF.")
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise UploadRejected(415, "File is not a readable PDF.")
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise UploadRejected(415, "Encrypted or password-protected PDFs are not accepted.")
        if doc.page_count > settings.max_pdf_pages:
            raise UploadRejected(413, f"PDF has {doc.page_count} pages (limit {settings.max_pdf_pages}).")
    finally:
        doc.close()


def _check_docx(data: bytes) -> None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise UploadRejected(415, "File is not a valid DOCX (corrupt ZIP container).")
    infos = zf.infolist()
    names = {i.filename for i in infos}
    if "[Content_Types].xml" not in names or "word/document.xml" not in names:
        raise UploadRejected(415, "ZIP file is not a Word document (only .docx is accepted).")
    if len(infos) > MAX_DOCX_ENTRIES:
        raise UploadRejected(415, "DOCX has too many internal parts.")
    total = sum(i.file_size for i in infos)
    if total > MAX_DOCX_UNCOMPRESSED or total > MAX_ZIP_RATIO * max(len(data), 1):
        raise UploadRejected(415, "DOCX expands to an unreasonable size (possible zip bomb).")
    content_types = zf.read("[Content_Types].xml").decode("utf-8", "ignore")
    if "macroEnabled" in content_types or any(n.lower().endswith("vbaproject.bin") for n in names):
        raise UploadRejected(415, "Macro-enabled Word documents are not accepted.")


def _check_image(data: bytes) -> None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > settings.max_image_pixels:
                raise UploadRejected(413, f"Image is {im.width}x{im.height} pixels (limit {settings.max_image_pixels // 1_000_000} MP).")
            im.verify()
    except UploadRejected:
        raise
    except Exception:  # includes PIL's DecompressionBombError
        raise UploadRejected(415, "Image is corrupt or too large to decode safely.")


def scan_for_malware(data: bytes) -> str:
    """clamd INSTREAM scan when CLAMAV_HOST is set; otherwise 'not_scanned' (recorded on the version)."""
    if not settings.clamav_host:
        return "not_scanned"
    try:
        with socket.create_connection((settings.clamav_host, settings.clamav_port), timeout=15) as s:
            s.sendall(b"zINSTREAM\0")
            for i in range(0, len(data), 1 << 16):
                chunk = data[i: i + (1 << 16)]
                s.sendall(struct.pack(">I", len(chunk)) + chunk)
            s.sendall(struct.pack(">I", 0))
            reply = s.recv(4096).decode("utf-8", "ignore")
    except OSError as e:
        raise UploadRejected(503, f"Malware scanner unavailable ({type(e).__name__}); upload refused.")  # fail closed
    if "FOUND" in reply:
        raise UploadRejected(415, "Upload rejected by the malware scanner.")
    return "clean"


def validate_upload(data: bytes) -> ValidatedUpload:
    if not data:
        raise UploadRejected(400, "Empty file.")
    if len(data) > max_bytes():
        raise UploadRejected(413, f"File is larger than {settings.max_upload_mb:g} MB.")
    mime = sniff(data)
    if mime is None:
        raise UploadRejected(415, "Unsupported file type. Accepted: PDF, DOCX, plain text, PNG, JPEG, TIFF.")
    if mime == PDF:
        _check_pdf(data)
    elif mime == DOCX:
        _check_docx(data)
    elif mime in IMAGES:
        _check_image(data)
    return ValidatedUpload(data, mime, scan_for_malware(data))
