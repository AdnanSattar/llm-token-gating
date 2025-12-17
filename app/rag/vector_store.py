from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.rag.embeddings import embed_query, embed_texts


class ChromaVectorStore:
    """
    Minimal ChromaDB wrapper for similarity search over text chunks.
    """

    def __init__(self, collection_name: str = "documents") -> None:
        settings = get_settings()
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.Client(
            ChromaSettings(
                is_persistent=True,
                anonymized_telemetry=False,
                persist_directory=str(persist_dir),
            )
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_texts(
        self, texts: Sequence[str], metadatas: Sequence[dict] | None = None
    ) -> None:
        if not texts:
            return
        embeddings = embed_texts(list(texts))
        ids = [f"doc-{self._collection.count()}-{i}" for i in range(len(texts))]
        self._collection.add(
            ids=ids,
            documents=list(texts),
            embeddings=embeddings,
            metadatas=list(metadatas) if metadatas is not None else None,
        )
        # persist
        self._client.persist()

    def similarity_search(self, query: str, k: int = 4) -> List[str]:
        if not query:
            return []
        query_embedding = embed_query(query)
        if not query_embedding:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        docs: Iterable[str] = results.get("documents", [[]])[0]
        return list(docs)


_VECTOR_STORE: ChromaVectorStore | None = None


def get_vector_store() -> ChromaVectorStore:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = ChromaVectorStore()
    return _VECTOR_STORE
