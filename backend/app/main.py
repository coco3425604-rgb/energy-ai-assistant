from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from backend.app.services.document_upload import (
    EmptyFileError,
    FileTooLargeError,
    InvalidPdfError,
    save_pdf,
)
from backend.app.services.pdf_text_extraction import (
    DocumentNotFoundError,
    InvalidDocumentFilenameError,
    PdfExtractionError,
    extract_pdf_text,
)

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"

app = FastAPI(title="AI Energy O&M Assistant API")


class UploadResult(BaseModel):
    success: bool
    message: str
    filename: str


class ExtractedPageResult(BaseModel):
    page_number: int
    text: str


class ExtractionResult(BaseModel):
    success: bool
    message: str
    filename: str
    page_count: int
    pages: list[ExtractedPageResult]


@app.post(
    "/api/documents/upload",
    response_model=UploadResult,
    status_code=status.HTTP_200_OK,
)
async def upload_document(file: UploadFile | None = File(default=None)) -> UploadResult:
    if file is None:
        raise HTTPException(status_code=400, detail="No file was provided")

    try:
        filename = await save_pdf(file, UPLOAD_DIR)
    except EmptyFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InvalidPdfError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file") from exc
    finally:
        await file.close()

    return UploadResult(
        success=True,
        message="PDF uploaded successfully",
        filename=filename,
    )


@app.post(
    "/api/documents/{filename}/extract",
    response_model=ExtractionResult,
    status_code=status.HTTP_200_OK,
)
def extract_document(filename: str) -> ExtractionResult:
    try:
        result = extract_pdf_text(filename, UPLOAD_DIR)
    except InvalidDocumentFilenameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PdfExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    message = (
        "PDF text extracted successfully"
        if result.has_extractable_text
        else "The PDF contains no extractable text layer"
    )
    return ExtractionResult(
        success=True,
        message=message,
        filename=result.filename,
        page_count=result.page_count,
        pages=[
            ExtractedPageResult(page_number=page.page_number, text=page.text)
            for page in result.pages
        ],
    )
