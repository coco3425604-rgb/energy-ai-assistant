import os
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

MAX_FILE_SIZE = 20 * 1024 * 1024
PDF_SIGNATURE = b"%PDF-"
CHUNK_SIZE = 1024 * 1024


class EmptyFileError(ValueError):
    pass


class InvalidPdfError(ValueError):
    pass


class FileTooLargeError(ValueError):
    pass


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "document.pdf").name
    return name if name not in {"", ".", ".."} else "document.pdf"


async def save_pdf(upload: UploadFile, upload_dir: Path) -> str:
    """Validate and atomically save one uploaded PDF."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{uuid4().hex}_{_safe_filename(upload.filename)}"
    destination = upload_dir / stored_filename
    temporary = upload_dir / f".{stored_filename}.part"
    total_size = 0
    first_chunk = True

    try:
        with temporary.open("xb") as output:
            while chunk := await upload.read(CHUNK_SIZE):
                if first_chunk:
                    first_chunk = False
                    if not chunk.startswith(PDF_SIGNATURE):
                        raise InvalidPdfError("The uploaded file is not a valid PDF")

                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise FileTooLargeError("The uploaded file exceeds the 20 MB limit")
                output.write(chunk)

        if total_size == 0:
            raise EmptyFileError("The uploaded file is empty")

        os.replace(temporary, destination)
        return stored_filename
    finally:
        temporary.unlink(missing_ok=True)
