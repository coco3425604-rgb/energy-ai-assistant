import hashlib

import pytest

from backend.app.services.chunk_embedding import (
    ChunkEmbeddingService,
    EmbeddingConfig,
)
from backend.app.services.text_chunking import TextChunk


class FakeEncoder:
    def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        return [
            [byte / 255.0 for byte in hashlib.sha256(text.encode()).digest()[:8]]
            for text in sentences
        ]


def _chunk(text: str, chunk_id: str = "manual.pdf:chunk:1") -> TextChunk:
    return TextChunk(chunk_id, "manual.pdf", 2, 3, text)


def _service(calls: list[tuple[str, dict[str, object]]] | None = None):
    def loader(model_name: str, **kwargs: object) -> FakeEncoder:
        if calls is not None:
            calls.append((model_name, kwargs))
        return FakeEncoder()

    return ChunkEmbeddingService(model_loader=loader)


def test_embeds_single_chunk_and_preserves_metadata() -> None:
    source = _chunk("变压器温度过高。")

    result = _service().embed_chunk(source)

    assert (result.chunk_id, result.filename) == (source.chunk_id, source.filename)
    assert (result.start_page, result.end_page, result.text) == (2, 3, source.text)
    assert len(result.embedding) == 8
    assert all(isinstance(value, float) for value in result.embedding)


def test_embeds_batch_with_fixed_dimension() -> None:
    results = _service().embed_chunks([_chunk("文本一"), _chunk("文本二", "c2")])

    assert len(results) == 2
    assert {len(result.embedding) for result in results} == {8}
    assert results[0].embedding != results[1].embedding


def test_same_text_has_stable_embedding() -> None:
    service = _service()
    assert service.embed_chunk(_chunk("相同文本")).embedding == service.embed_chunk(
        _chunk("相同文本", "c2")
    ).embedding


@pytest.mark.parametrize("text", ["", "   \n"])
def test_rejects_empty_chunk_text(text: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _service().embed_chunk(_chunk(text))


def test_empty_batch_does_not_load_model() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    assert _service(calls).embed_chunks([]) == []
    assert calls == []


def test_model_configuration_and_loading_are_replaceable() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    config = EmbeddingConfig(
        model_name="local/test-model",
        cache_dir="outside-repository-cache",
        device="cpu",
        local_files_only=True,
    )
    service = ChunkEmbeddingService(config=config, model_loader=_service(calls)._model_loader)

    service.embed_chunk(_chunk("配置测试"))

    assert calls == [
        (
            "local/test-model",
            {
                "cache_folder": "outside-repository-cache",
                "device": "cpu",
                "local_files_only": True,
            },
        )
    ]
