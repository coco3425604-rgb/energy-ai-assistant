from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
from typing import Any, Protocol, cast

from backend.app.services.text_chunking import TextChunk


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


ModelLoader = Callable[..., Encoder]


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    cache_dir: str | None = None
    device: str | None = None
    local_files_only: bool = False

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL),
            cache_dir=os.getenv("EMBEDDING_CACHE_DIR") or None,
            device=os.getenv("EMBEDDING_DEVICE") or None,
            local_files_only=os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "false").lower()
            in {"1", "true", "yes"},
        )


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk_id: str
    filename: str
    start_page: int
    end_page: int
    text: str
    embedding: list[float]


def _default_model_loader(model_name: str, **kwargs: Any) -> Encoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - installation error only
        raise RuntimeError(
            "sentence-transformers is required for local embeddings"
        ) from exc
    return cast(Encoder, SentenceTransformer(model_name, **kwargs))


class ChunkEmbeddingService:
    """Generate local, normalized embeddings while retaining chunk provenance."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig.from_env()
        self._model_loader = model_loader or _default_model_loader
        self._model: Encoder | None = None
        self._dimension: int | None = None

    def _load_model(self) -> Encoder:
        if self._model is None:
            options: dict[str, Any] = {
                "local_files_only": self.config.local_files_only,
            }
            if self.config.cache_dir is not None:
                options["cache_folder"] = self.config.cache_dir
            if self.config.device is not None:
                options["device"] = self.config.device
            self._model = self._model_loader(self.config.model_name, **options)
        return self._model

    def embed_chunk(self, chunk: TextChunk) -> EmbeddedChunk:
        return self.embed_chunks([chunk])[0]

    def embed_chunks(self, chunks: Sequence[TextChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []
        if any(not chunk.text.strip() for chunk in chunks):
            raise ValueError("chunk text must not be empty")

        # E5 retrieval models expect the passage prefix for corpus documents.
        inputs = [f"passage: {chunk.text}" for chunk in chunks]
        raw_vectors = self._load_model().encode(
            inputs,
            batch_size=len(inputs),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = [[float(value) for value in vector] for vector in raw_vectors]
        if len(vectors) != len(chunks) or any(not vector for vector in vectors):
            raise RuntimeError("embedding model returned an invalid result")

        dimensions = {len(vector) for vector in vectors}
        if self._dimension is not None:
            dimensions.add(self._dimension)
        if len(dimensions) != 1:
            raise RuntimeError("embedding model returned inconsistent dimensions")
        self._dimension = len(vectors[0])

        return [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                text=chunk.text,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
