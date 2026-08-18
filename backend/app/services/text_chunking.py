from dataclasses import dataclass
import re
from typing import Sequence

from backend.app.services.pdf_text_extraction import ExtractedPage


DEFAULT_MAX_LENGTH = 800
DEFAULT_OVERLAP = 120

_PARAGRAPH_BOUNDARY = re.compile(r"\n[ \t]*\n+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])(?:[ \t]*|\n+)|(?<=[.])(?:[ \t]+|\n+)")
_WHITESPACE_BOUNDARY = re.compile(r"\s+")


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    filename: str
    start_page: int
    end_page: int
    text: str


def _build_document(pages: Sequence[ExtractedPage]) -> tuple[str, list[int]]:
    """Build an ordered text stream and retain the source page for every character."""
    parts: list[str] = []
    page_map: list[int] = []

    for page in pages:
        text = page.text.strip()
        if not text:
            continue

        if parts:
            # A page break is a weak structural boundary. Two newlines allow a
            # chunk to cross it while still making it preferable to cut there.
            parts.append("\n\n")
            page_map.extend([page.page_number, page.page_number])

        parts.append(text)
        page_map.extend([page.page_number] * len(text))

    return "".join(parts), page_map


def _boundary_positions(pattern: re.Pattern[str], text: str) -> list[int]:
    return [match.end() for match in pattern.finditer(text)]


def _choose_end(
    start: int,
    limit: int,
    paragraph_ends: Sequence[int],
    sentence_ends: Sequence[int],
    whitespace_ends: Sequence[int],
) -> int:
    """Choose the strongest useful boundary without producing a tiny chunk."""
    minimum_useful = start + max(1, (limit - start) // 2)
    for boundaries in (paragraph_ends, sentence_ends, whitespace_ends):
        candidates = [position for position in boundaries if minimum_useful <= position <= limit]
        if candidates:
            return candidates[-1]
    return limit


def _choose_next_start(
    current_start: int,
    end: int,
    overlap: int,
    paragraph_ends: Sequence[int],
    sentence_ends: Sequence[int],
) -> int:
    if overlap == 0:
        return end

    desired = max(current_start + 1, end - overlap)
    # Starting immediately after a natural boundary avoids cutting a sentence
    # merely to achieve an exact overlap size. Prefer the closest such boundary.
    natural_boundaries = sorted(
        {
            position
            for position in (*paragraph_ends, *sentence_ends)
            if desired <= position < end
        }
    )
    return natural_boundaries[0] if natural_boundaries else desired


def chunk_pages(
    filename: str,
    pages: Sequence[ExtractedPage],
    max_length: int = DEFAULT_MAX_LENGTH,
    overlap: int = DEFAULT_OVERLAP,
) -> list[TextChunk]:
    """Split ordered PDF pages into structure-aware chunks with page provenance."""
    if not filename:
        raise ValueError("filename must not be empty")
    if max_length <= 0:
        raise ValueError("max_length must be greater than zero")
    if overlap < 0 or overlap >= max_length:
        raise ValueError("overlap must be non-negative and smaller than max_length")

    document, page_map = _build_document(pages)
    if not document:
        return []

    paragraph_ends = _boundary_positions(_PARAGRAPH_BOUNDARY, document)
    sentence_ends = _boundary_positions(_SENTENCE_BOUNDARY, document)
    whitespace_ends = _boundary_positions(_WHITESPACE_BOUNDARY, document)

    chunks: list[TextChunk] = []
    start = 0
    while start < len(document):
        limit = min(start + max_length, len(document))
        end = (
            len(document)
            if limit == len(document)
            else _choose_end(
                start,
                limit,
                paragraph_ends,
                sentence_ends,
                whitespace_ends,
            )
        )

        chunk_text = document[start:end].strip()
        if chunk_text:
            content_start = start
            content_end = end - 1
            while content_start < end and document[content_start].isspace():
                content_start += 1
            while content_end >= content_start and document[content_end].isspace():
                content_end -= 1

            chunks.append(
                TextChunk(
                    chunk_id=f"{filename}:chunk:{len(chunks) + 1}",
                    filename=filename,
                    start_page=page_map[content_start],
                    end_page=page_map[content_end],
                    text=chunk_text,
                )
            )

        if end == len(document):
            break
        start = _choose_next_start(
            start,
            end,
            overlap,
            paragraph_ends,
            sentence_ends,
        )

    return chunks
