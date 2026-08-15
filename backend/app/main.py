from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from backend.app.services.document_upload import (
    EmptyFileError,
    FileTooLargeError,
    InvalidPdfError,
    save_pdf,
)

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"

app = FastAPI(title="AI Energy O&M Assistant API")


class UploadResult(BaseModel):
    success: bool
    message: str
    filename: str


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
