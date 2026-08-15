from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from backend.app import main
from backend.app.services.document_upload import MAX_FILE_SIZE


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    return TestClient(main.app)


def test_uploads_pdf(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("manual.pdf", b"%PDF-1.7\nmock pdf content", "application/pdf")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["message"] == "PDF uploaded successfully"
    assert result["filename"].endswith("_manual.pdf")
    assert (tmp_path / result["filename"]).read_bytes().startswith(b"%PDF-")


def test_rejects_missing_file(client: TestClient) -> None:
    response = client.post("/api/documents/upload")

    assert response.status_code == 400


def test_rejects_non_pdf_file(client: TestClient) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 400


def test_rejects_non_pdf_disguised_with_pdf_extension(client: TestClient) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("fake.pdf", b"not actually a pdf", "application/pdf")},
    )

    assert response.status_code == 400


def test_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400


def test_rejects_file_larger_than_20_mb(client: TestClient) -> None:
    oversized_pdf = b"%PDF-" + b"0" * (MAX_FILE_SIZE - len(b"%PDF-") + 1)
    response = client.post(
        "/api/documents/upload",
        files={"file": ("large.pdf", oversized_pdf, "application/pdf")},
    )

    assert response.status_code == 413
