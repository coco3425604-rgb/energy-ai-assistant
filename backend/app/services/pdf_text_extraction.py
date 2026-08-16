from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class InvalidDocumentFilenameError(ValueError):
    pass


class DocumentNotFoundError(FileNotFoundError):
    pass


class PdfExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class PdfExtractionResult:
    filename: str
    page_count: int
    pages: list[ExtractedPage]
    has_extractable_text: bool


def _resolve_uploaded_pdf(filename: str, upload_dir: Path) -> Path:
    """Resolve a stored PDF name without allowing access outside upload_dir."""
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
        or Path(filename).suffix.lower() != ".pdf"
    ):
        raise InvalidDocumentFilenameError("Invalid PDF filename")

    allowed_dir = upload_dir.resolve()
    candidate = (allowed_dir / filename).resolve()
    if not candidate.is_relative_to(allowed_dir):
        raise InvalidDocumentFilenameError("Invalid PDF filename")
    if not candidate.is_file():
        raise DocumentNotFoundError(f"PDF '{filename}' was not found")

    return candidate


def extract_pdf_text(filename: str, upload_dir: Path) -> PdfExtractionResult:
    pdf_path = _resolve_uploaded_pdf(filename, upload_dir)

    try:
        reader = PdfReader(pdf_path, strict=True)
        pages = [
            ExtractedPage(page_number=index, text=page.extract_text() or "")
            for index, page in enumerate(reader.pages, start=1)
        ]
    except (PdfReadError, OSError, ValueError, TypeError, KeyError) as exc:
        raise PdfExtractionError("The PDF is damaged or cannot be parsed") from exc

    return PdfExtractionResult(
        filename=filename,
        page_count=len(pages),
        pages=pages,
        has_extractable_text=any(page.text.strip() for page in pages),
    )
