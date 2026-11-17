"""
Section-aware recursive chunker with overlap.

1. Blocks are grouped into sections at every heading (text before the first heading is section "").
2. Each section is packed into chunks of ~target characters, splitting long blocks recursively:
   paragraph -> sentences -> words. Chunks NEVER span two sections, so a chunk about "Allergies" never
   carries the tail of "Medications".
3. Consecutive chunks in the same section overlap by ~overlap characters (whole sentences where possible),
   so a fact sitting on a boundary is retrievable from either side.
4. Every chunk keeps provenance: section title and index, page range, character offsets into the
   document's extracted text (ParsedDocument.text).

The chunker id (name + parameters) is stored on each version; changing it marks versions stale for re-index.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from backend.app.ingestion.parsers import Block, ParsedDocument

CHUNKER_NAME = "section-recursive-v1"


@dataclass
class Chunk:
    index: int
    text: str
    section: str
    section_index: int
    page_start: Optional[int]
    page_end: Optional[int]
    char_start: int
    char_end: int


def chunker_id(target: int, overlap: int) -> str:
    return f"{CHUNKER_NAME}:{target}/{overlap}"


_SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\-•*])|\n+")


def _pieces(text: str, target: int) -> List[str]:
    """Recursive split into pieces <= target: keep whole if it fits, else sentences, else words."""
    if len(text) <= target:
        return [text]
    sentences = [s.strip() for s in _SENT.split(text) if s.strip()]
    if len(sentences) > 1:
        out: List[str] = []
        for s in sentences:
            out.extend(_pieces(s, target))
        return out
    words, out, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > target:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def _sections(blocks: List[Block]) -> List[Tuple[str, List[Tuple[Block, int]]]]:
    """[(title, [(block, char_offset_in_document_text)])], offsets matching ParsedDocument.text."""
    sections: List[Tuple[str, List[Tuple[Block, int]]]] = [("", [])]
    offset = 0
    for b in blocks:
        if b.kind == "heading":
            sections.append((b.text, []))
        else:
            sections[-1][1].append((b, offset))
        offset += len(b.text) + 2  # ParsedDocument.text joins blocks with "\n\n"
    return [s for s in sections if s[1]]


def chunk_document(doc: ParsedDocument, target: int = 900, overlap: int = 150) -> List[Chunk]:
    chunks: List[Chunk] = []
    for s_idx, (title, items) in enumerate(_sections(doc.blocks)):
        # (piece_text, page, char_start) units for this section
        units: List[Tuple[str, Optional[int], int]] = []
        for block, start in items:
            pos = 0
            for piece in _pieces(block.text, target):
                found = block.text.find(piece, pos)
                at = start + (found if found >= 0 else pos)
                pos = (found if found >= 0 else pos) + len(piece)
                units.append((piece, block.page, at))

        i = 0
        while i < len(units):
            j, size = i, 0
            while j < len(units) and (j == i or size + 1 + len(units[j][0]) <= target):
                size += len(units[j][0]) + (1 if j > i else 0)
                j += 1
            group = units[i:j]
            pages = [u[1] for u in group if u[1] is not None]
            last = group[-1]
            chunks.append(Chunk(
                index=len(chunks), text=" ".join(u[0] for u in group), section=title, section_index=s_idx,
                page_start=min(pages) if pages else None, page_end=max(pages) if pages else None,
                char_start=group[0][2], char_end=last[2] + len(last[0]),
            ))
            if j >= len(units):
                break
            # overlap: step back over trailing units totalling <= overlap chars, but always advance
            back, k = 0, j
            while k - 1 > i and back + len(units[k - 1][0]) <= overlap:
                back += len(units[k - 1][0]) + 1
                k -= 1
            i = k if k > i else j
    return chunks
