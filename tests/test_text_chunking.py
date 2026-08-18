import re

import pytest

from backend.app.services.pdf_text_extraction import ExtractedPage
from backend.app.services.text_chunking import chunk_pages


def _pages(*texts: str) -> list[ExtractedPage]:
    return [
        ExtractedPage(page_number=number, text=text)
        for number, text in enumerate(texts, start=1)
    ]


def test_single_short_page_produces_one_chunk() -> None:
    chunks = chunk_pages("manual.pdf", _pages("短文本。"))

    assert len(chunks) == 1
    assert chunks[0].filename == "manual.pdf"
    assert chunks[0].start_page == chunks[0].end_page == 1
    assert chunks[0].text == "短文本。"


def test_multiple_paragraphs_are_split_at_paragraph_boundaries() -> None:
    paragraphs = ["A" * 30, "B" * 30, "C" * 30]
    chunks = chunk_pages(
        "manual.pdf", _pages("\n\n".join(paragraphs)), max_length=65, overlap=0
    )

    assert chunks[0].text == f"{paragraphs[0]}\n\n{paragraphs[1]}"
    assert chunks[1].text == paragraphs[2]


def test_long_paragraph_prefers_sentence_boundaries() -> None:
    text = "第一句内容很完整。第二句内容也很完整。第三句仍然保持完整。"
    chunks = chunk_pages("manual.pdf", _pages(text), max_length=24, overlap=0)

    assert len(chunks) > 1
    assert all(chunk.text.endswith("。") for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def test_very_long_unbroken_text_respects_max_length() -> None:
    chunks = chunk_pages("manual.pdf", _pages("x" * 205), max_length=50, overlap=10)

    assert all(len(chunk.text) <= 50 for chunk in chunks)
    assert chunks[0].text == "x" * 50
    assert chunks[1].text.startswith("x" * 10)


def test_multiple_pages_can_form_cross_page_chunks_with_correct_ranges() -> None:
    chunks = chunk_pages(
        "manual.pdf",
        _pages("第一页内容。", "第二页内容。", "第三页内容。"),
        max_length=18,
        overlap=0,
    )

    assert [(chunk.start_page, chunk.end_page) for chunk in chunks] == [(1, 2), (3, 3)]
    assert chunks[0].text == "第一页内容。\n\n第二页内容。"
    assert chunks[1].text == "第三页内容。"


def test_chunks_preserve_source_order() -> None:
    chunks = chunk_pages(
        "manual.pdf", _pages("one. two. three. four."), max_length=12, overlap=0
    )

    reconstructed = "".join(chunk.text for chunk in chunks)
    assert re.sub(r"\s+", "", reconstructed) == re.sub(
        r"\s+", "", "one. two. three. four."
    )


def test_chunk_ids_are_unique() -> None:
    chunks = chunk_pages("manual.pdf", _pages("句子。" * 20), max_length=20, overlap=4)

    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_empty_pages_do_not_create_chunks_or_incorrect_page_ranges() -> None:
    chunks = chunk_pages("manual.pdf", _pages("", "  \n", "有效文本。", ""))

    assert len(chunks) == 1
    assert chunks[0].start_page == chunks[0].end_page == 3
    assert chunk_pages("empty.pdf", _pages("", "\n")) == []


def test_overlap_reuses_ordered_suffix_without_duplicate_chunks() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = chunk_pages("manual.pdf", _pages(text), max_length=24, overlap=6)

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        shared_lengths = [
            size
            for size in range(1, min(6, len(previous.text), len(current.text)) + 1)
            if previous.text[-size:] == current.text[:size]
        ]
        assert shared_lengths
        assert previous.text != current.text


@pytest.mark.parametrize(
    ("max_length", "overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_rejects_invalid_length_configuration(max_length: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_pages("manual.pdf", _pages("text"), max_length=max_length, overlap=overlap)
