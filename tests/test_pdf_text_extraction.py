from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pytest

from backend.app import main


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    return TestClient(main.app)


def _write_pdf(path: Path, page_texts: list[str | None]) -> None:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)

    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
        )

    with path.open("wb") as output:
        writer.write(output)


def test_extracts_text_from_multiple_pages_in_order(
    client: TestClient, tmp_path: Path
) -> None:
    _write_pdf(tmp_path / "manual.pdf", ["First page", "Second page", "Third page"])

    response = client.post("/api/documents/manual.pdf/extract")

    assert response.status_code == 200
    result = response.json()
    assert result == {
        "success": True,
        "message": "PDF text extracted successfully",
        "filename": "manual.pdf",
        "page_count": 3,
        "pages": [
            {"page_number": 1, "text": "First page"},
            {"page_number": 2, "text": "Second page"},
            {"page_number": 3, "text": "Third page"},
        ],
    }


def test_returns_404_when_pdf_does_not_exist(client: TestClient) -> None:
    response = client.post("/api/documents/missing.pdf/extract")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_returns_clear_error_for_damaged_pdf(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "damaged.pdf").write_bytes(b"%PDF-1.7\nnot a real PDF")

    response = client.post("/api/documents/damaged.pdf/extract")

    assert response.status_code == 422
    assert response.json()["detail"] == "The PDF is damaged or cannot be parsed"


def test_reports_pdf_without_extractable_text(client: TestClient, tmp_path: Path) -> None:
    _write_pdf(tmp_path / "scanned.pdf", [None, None])

    response = client.post("/api/documents/scanned.pdf/extract")

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["message"] == "The PDF contains no extractable text layer"
    assert result["page_count"] == 2
    assert result["pages"] == [
        {"page_number": 1, "text": ""},
        {"page_number": 2, "text": ""},
    ]


@pytest.mark.parametrize(
    "url",
    [
        "/api/documents/../secret.pdf/extract",
        "/api/documents/%2E%2E%2Fsecret.pdf/extract",
        "/api/documents/%2E%2E%5Csecret.pdf/extract",
        "/api/documents/not-a-pdf.txt/extract",
    ],
)
def test_rejects_invalid_or_traversal_filenames(client: TestClient, url: str) -> None:
    response = client.post(url)

    assert response.status_code in {400, 404}
